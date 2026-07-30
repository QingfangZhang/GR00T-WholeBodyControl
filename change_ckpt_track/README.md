# qpos-track：在原任务场景中比较 regular 与 low-latency

这套工具把原 recording 中 regular checkpoint **实际走过的机器人状态**
转换成 50 Hz 身体 reference，并在同一份任务场景、同一份身体 reference 和
同一份手指 target 下，分别闭环运行 regular 与 low-latency checkpoint。

原项目目录和 `change_ckpt/` 都不会被修改。新增代码位于
`change_ckpt_track/`，持久化运行结果位于 `change_ckpt_track/data/`。

数据流是：

```text
原 data.csv
  ├─ 每个 policy_seq 的第一行 body qpos/qvel 和 pelvis quaternion
  ├─ 同一帧的 left/right hand target
  └─ 原 recording 的 MuJoCo 场景和任务物体初始状态
          ↓
50 Hz qpos reference → C++ 本地 encoder/decoder → DDS 命令
          ↓
带任务物体的 MuJoCo 闭环仿真 → 新 data.csv + deploy CSV + 终态 metadata
```

默认 `--root-assist none` 不会逐帧覆盖新仿真的机器人 qpos，也不会逐帧覆盖
任务物体 qpos。只有显式启用下文的 root-assist 诊断时，simulator 才逐步硬对齐
选中的 floating-base 平移分量；身体/手关节和任务物体仍不会被覆盖。任务物体
只从 recording 恢复初始状态，之后由新一轮接触动力学推进。

## 这个实验能回答什么

它主要回答：

> 面对 regular checkpoint 已成功（或实际）走过的同一条机器人身体轨迹，
> low-latency checkpoint 能否稳定跟踪，并维持相同的手和物体交互？

这条 reference 本身来自 regular 的实际 qpos，所以对 regular 存在来源优势。
它是很有用的诊断实验，但不能单独替代“原始人体 retarget reference 下更换
checkpoint”的正式实验，也不能只凭一个 RMSE 数值认定语义任务成功。最终仍要
结合实时 viewer 或 replay 检查物体交互结果。

## reference 是怎样生成的

默认处理规则：

- 以连续的 `policy_seq` 分组，每组取第一行，得到约 50 Hz 的策略帧；
- 若首尾 `policy_seq` 的行数不到典型组的一半，视为录制截断并丢弃；
- 从 MuJoCo qpos/qvel 中按关节名提取 29 个身体关节，排除 14 个手指关节和
  任务物体自由度；
- 29 维身体状态转换为 G1 IsaacLab 顺序后发给 deploy；
- pelvis 世界位置和 `w,x,y,z` 四元数都从同一 qpos 帧提取；
- protocol v1 实时链路只发送四元数，不发送 `body_pos`；pelvis 世界位置仅保留
  在导出的 `body_pos.csv`、`prepared_reference.npz` 和诊断文件中；
- 左右各 7 维手指 target 与身体帧同步保留，两套 checkpoint 收到完全相同的
  外部手 target；
- 默认不读取原 CSV 的 `reference_motion`，也从不发送原 CSV 的
  `token_state`。只有下文的 `--regular-future-window recorded` 实验会读取
  `reference_motion`，且仅用于推断十个槽的时间 lag；实际发送的数值仍全部来自
  qpos/qvel/pelvis quaternion。

publisher 默认每次发送 100 帧连续身体 reference。regular encoder 的最远
前视是当前帧之后 45 帧；100 帧首包能在 CONTROL 启动和下一包合并之间保留
足够余量，避免 50 帧首包曾出现的单周期边界 hold。比较器仍会逐帧审计
duplicate/skip，不能仅凭配置假定播放严格同步。

手控制与 29 维身体策略是两条不同链路：左右手 target 绕过 body
encoder/decoder，直接作为外部手命令。它还是 **latest-value** 语义——每个
publisher tick 只发送当前各 7 维 target，而不是像身体 reference 那样在 C++
`MotionSequence` 中维护未来 frame buffer。因此同一手 target 不等于两次实验的
手一定走出相同状态；需要同时核验 `left/right_hand_action.csv` 和对应的
`left/right_hand_q.csv`。

统一 launcher 会在内存中完成这一步，并保存
`reference_diagnostics.json` 和 `prepared_reference.npz`。如果想先导出成
项目标准的 deploy reference 文件检查，可以单独运行：

```bash
source .venv_sim/bin/activate

python change_ckpt_track/qpos_reference_data.py \
  sample_data/ztj/20260612/20260612_144117_g1_sim \
  --output-dir change_ckpt_track/reference/20260612_144117_g1_sim \
  --motion-name 20260612_144117_g1_sim_qpos
```

生成：

```text
change_ckpt_track/reference/20260612_144117_g1_sim/
└── 20260612_144117_g1_sim_qpos/
    ├── joint_pos.csv          # 29维，IsaacLab顺序，rad
    ├── joint_vel.csv          # 29维，IsaacLab顺序，rad/s
    ├── body_pos.csv           # pelvis世界位置
    ├── body_quat.csv          # pelvis四元数，w,x,y,z
    ├── metadata.txt
    ├── frame_map.csv          # 输出帧到源CSV行/policy_seq的映射
    └── conversion_report.json # 列映射、范围和数值诊断
```

只检查导出的身体 reference 时，可以使用项目的 `visualize_motion.py`，但它要求
**同一个 Python 环境同时安装 MuJoCo、lxml、SciPy 和脚本的其他依赖**。当前
`.venv_sim` 缺少 lxml，而 isaaclab 环境缺少 MuJoCo，因此下面的命令在这两个
现有环境中都不能直接运行；准备好满足依赖的环境后才可执行：

```bash
cd gear_sonic_deploy
python visualize_motion.py \
  --motion_dir ../change_ckpt_track/reference/20260612_144117_g1_sim/20260612_144117_g1_sim_qpos
cd ..
```

这个 reference visualizer 是空场景检查工具；真正运行下面的 rollout 时才会加载
recording 中的任务场景和物体。

还有一个当前 streamed protocol v1 的限制：虽然导出的 `body_pos.csv` 包含原
qpos pelvis 位置，publisher 不会把它传给 C++。streamed `MotionSequence` 的
`BodyPositions` 当前保持默认值（通常是零）。不过 C++ 的 mode filter 下，当前
regular 和 low-latency 的 G1（mode 0）`required_observations` 都只有
`encoder_mode`、29 关节位置/速度和 anchor orientation，不包含 root-z。因此，
本次实际比较的 G1 640 维运动输入和 29 关节 tracking 指标不受 `body_pos`
缺失影响。这个限制只影响非 G1 模式或完整 observation superset 的解释，并会
记录在诊断和 manifest 中。

## 1. 运行前检查

从项目根目录、在 `.venv_sim` 环境运行。先确保没有旧的 MuJoCo simulator、
SONIC deploy 或 publisher 占用 DDS/ZMQ：

```bash
source .venv_sim/bin/activate

python change_ckpt_track/launch_checkpoint_rollout.py preflight \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260612_144117_g1_sim

python change_ckpt_track/launch_checkpoint_rollout.py preflight \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/20260612_144117_g1_sim
```

默认模型是：

```text
regular:
  change_ckpt/models/regular/model_encoder.onnx
  change_ckpt/models/regular/model_decoder.onnx
  change_ckpt/observation_config_sonic_release.yaml

low_latency:
  change_ckpt/models/low_latency/model_encoder.onnx
  change_ckpt/models/low_latency/model_decoder.onnx
  change_ckpt/models/low_latency/observation_config.yaml
```

如需使用其他文件，给 `preflight` 和 `run` 同时传
`--encoder`、`--decoder`、`--obs-config`。`launch_manifest.json` 会记录实际
模型路径和 SHA-256，之后不需要凭文件名猜测究竟跑了哪个 checkpoint。

## 2. 分别实时运行两套 checkpoint

一次只运行一套。launcher 会统一启动任务 simulator、C++ deploy 和 qpos
reference publisher；默认打开 MuJoCo viewer、自动完成 INIT/CONTROL/reference
播放，到 reference 结束后停止，并默认保存 CSV。

regular：

```bash
python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260612_144117_g1_sim
```

low-latency：

```bash
python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/20260612_144117_g1_sim
```

运行时直接在 viewer 中观察：

- 机器人是否跌倒、抖动、滑步或明显落后于动作；
- 手指是否按原 target 开合；
- 手是否真正接触任务物体；
- 抽屉、scanner 或其他物体是否沿预期方向运动；
- regular 与 low-latency 在相同动作阶段是否出现不同结果。

正式定量结果建议再各跑一次 `--no-viewer`，避免 GUI 刷新影响实时性：

```bash
python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260612_144117_g1_sim \
  --no-viewer

python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/20260612_144117_g1_sim \
  --no-viewer
```

输出目录固定为：

```text
change_ckpt_track/data/20260612_144117_g1_sim_regular/
change_ckpt_track/data/20260612_144117_g1_sim_low_latency/
```

同一 recording 和 checkpoint 重复运行会覆盖对应的同名目录，不使用时间戳。

### 可选：运行时补偿原 recording 的 root

为了检查 qpos-track 任务失败是否主要来自 root 平移漂移，可以启用与
`change_ckpt` 相同的 oracle root-assist。下面命令明确使用 normal/canonical
lag，并只补水平位置：

```bash
python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim \
  --regular-future-window canonical \
  --root-assist xy \
  --no-viewer

python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim \
  --regular-future-window canonical \
  --root-assist xy \
  --no-viewer
```

`canonical` 对 regular 表示 encoder future slot 使用
`[0,5,10,...,45]`；对 low-latency 仍是连续 `[0,1,...,9]`，两者都不会启用
recorded-lag 重排。root-assist 与 encoder slot lag 是两条独立时间链：
simulator 每个 200 Hz 物理周期结束后，从原始 400 Hz recording 的当前行硬写
`qpos[x,y]` 和 `qvel[vx,vy]`，再写 CSV 并进入下一周期。它不从构造的 50 Hz
future slot 取 root。

可选值为：

- `--root-assist none`：默认，无辅助；
- `--root-assist xy`：对齐 `x/y` 与 `vx/vy`，推荐先测；
- `--root-assist xyz`：额外对齐高度与竖直速度。

辅助输出与无辅助基线分开保存：

```text
change_ckpt_track/data/20260720_144342_g1_sim_regular_root_assist_xy/
change_ckpt_track/data/20260720_144342_g1_sim_low_latency_root_assist_xy/
```

同一 assisted 条件重复运行会覆盖自己的同名目录。`run_metadata.json` 会记录
补偿前后误差和耗时，`launch_manifest.json` 会记录 root-assist 与
canonical/recorded lag 模式。这个实验使用原成功轨迹的 root 真值并改变接触
动力学，只能作为诊断上限，不能算 checkpoint 无辅助完成任务。

正式生成必须使用 `--no-viewer`。当前 simulator 与 publisher 是独立墙钟线程，
同步 viewer 可能让 MuJoCo 低于实时，而 50 Hz reference 继续按墙钟前进，造成
动作相对 source/ghost 加速。生成结束后再用第5节 ghost replay 查看。

### regular 使用 recording 中的十槽时间布局

默认 `canonical` 模式发送连续 qpos，因此 regular C++ step-5 gatherer 读到
`[0,5,10,...,45]`。要测试原 recording 实际保存的槽时间（例如
`[0,5,9,9,...,9]`），使用：

```bash
python change_ckpt_track/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim \
  --regular-future-window recorded \
  --no-viewer
```

程序会对每个 recording 自动推断十个 lag，而不是统一写死一个数组。当前两段
测试数据分别推断为：

```text
20260720_144342_g1_sim  -> [0,5,9,9,9,9,9,9,9,9]
20260612_144127_g1_sim  -> [0,4,4,4,4,4,4,4,4,4]
```

publisher 同步重排 joint position、joint velocity 和 pelvis quaternion；
wire 上的 `frame_index` 仍连续。输出使用独立、可重复覆盖的目录，不会覆盖默认
regular 结果：

```text
change_ckpt_track/data/<recording>_regular_recorded_lags/
```

实际 lag、每个槽的推断误差和重排模式会同时写入
`reference_diagnostics.json`、`prepared_reference.npz` 和
`launch_manifest.json`。这个选项只允许 regular checkpoint；low-latency 仍使用
连续的 `[0,1,...,9]`。

## 3. 每次运行保存什么

```text
change_ckpt_track/data/<recording>_<checkpoint>/
├── data.csv
├── deploy_csv/
│   ├── q.csv
│   ├── dq.csv
│   ├── action.csv
│   ├── token_state.csv
│   ├── motion_playing.csv
│   ├── left_hand_q.csv
│   ├── right_hand_q.csv
│   └── ...
├── target_motion.csv
├── prepared_reference.npz
├── reference_diagnostics.json
├── run_metadata.json
├── launch_manifest.json
├── sim.log
├── deploy.log
└── publisher.log
```

关键含义：

- `data.csv`：本次带任务物体的 MuJoCo 完整状态，可再次 replay；
- `deploy_csv/q.csv`：本次实测 29 关节位置，MuJoCo/硬件顺序；
- `deploy_csv/dq.csv`：本次实测 29 关节速度，同一顺序；
- `deploy_csv/action.csv`：decoder 的 29 维 raw action，IsaacLab 顺序，既不是
  `q_target`，也没有乘 action scale；
- `deploy_csv/token_state.csv`：本次本地 encoder 真正生成的 token；
- `motion_playing.csv`：定量分析的播放区间依据；
- `left/right_hand_action.csv`：C++ 当时采用的外部手 target；它不经过 body
  encoder/decoder；
- `left/right_hand_q.csv`：仿真中实际测得的左右手关节位置，应与对应 action
  一起检查延迟、漏更新和跟随误差；
- `target_motion.csv`：C++ 本次实际消费的 reference，每行是
  `root xyz + root quat wxyz + 29 joint q(MuJoCo顺序)`；其中 29 关节和
  quaternion 来自本次 qpos reference，但 protocol v1 不传 `body_pos`，所以
  root xyz 通常是 C++ streamed motion 的默认零值，不能当作原 qpos pelvis
  世界位置；
- `run_metadata.json`：跌倒、非有限状态、实时因子、任务物体初态和终态；
- `launch_manifest.json`：实际模型、配置、reference 和启动命令；
- `reference_diagnostics.json`：qpos 列映射、源 CSV 行、`policy_seq`、首尾裁剪
  和数值检查。

## 4. 自动比较结果

两套运行完成后：

```bash
python change_ckpt_track/compare_qpos_track.py \
  change_ckpt_track/data/20260612_144117_g1_sim_regular \
  change_ckpt_track/data/20260612_144117_g1_sim_low_latency \
  --plot
```

默认读取 manifest 中的 `warmup_exclusion_s`；如果没有则排除播放开始后的
0.2 秒。要包含全部播放帧，可加 `--warmup-s 0`。

比较区间不是按文件开头盲目截取，而是：

1. 在各自 `motion_playing.csv` 中找到第一次 `playing > 0.5`；
2. 取从这里开始的第一个连续播放段；
3. 用 logger `index` 对齐 `q/dq/action/token` 与 `target_motion`；
4. 将 `target_motion` 的 29 关节逐行匹配回
   `prepared_reference.npz/joint_pos`，恢复对应的 reference frame 和
   `policy_seq`；匹配误差必须不超过 `1e-5 rad`；
5. 两套运行取共同 reference frame 的第一次出现进行比较，而不是假设两个独立
   50 Hz 循环的播放 ordinal 永远相同。非终止重复帧、跳帧和两边 ordinal
   schedule 是否一致会单独写入诊断，不会被静默隐藏。

结果写入：

```text
change_ckpt_track/data/20260612_144117_g1_sim_comparison/
├── summary.json
├── per_joint_metrics.csv
├── aligned_tracking.csv
└── tracking_overview.png
```

查看顺序建议：

1. 先看 `summary.json` 的 `runtime_validity`、`fallen`、finite 和
   `same_target_within_1e-6`，确认共同 reference frame 上 target 完全一致；
   同时检查 `same_reference_schedule_by_ordinal` 和两套
   `reference_frame_alignment`。如果前者为 false，说明某次运行发生过额外
   hold/duplicate；比较器会纠正统计对齐，但严格 A/B 仍建议重新运行；
2. 检查每套运行的 `protocol_timing_audit`：要求 publisher `late_ticks=0`，
   且首包初始化之后没有 `forcing catch-up`、额外 `did_catchup=1` 或
   `Catch-up: Reset`；首个 packet 的一次 `did_catchup=1 + Reset` 是 C++ 建立
   streamed motion 的正常行为，不会被比较器误判；
3. 看 overall 及左右腿、腰、左右手臂的 tracking RMSE/max；
4. 看 `per_joint_metrics.csv` 定位误差最大的关节；
5. 看 raw action 的 step-delta RMS 和最大跳变，判断输出是否更抖；
6. 对照两次运行的 `left/right_hand_action.csv` 与
   `left/right_hand_q.csv`；手是 latest-value 外部链路，不能用 body tracking
   RMSE 代替这项检查；
7. 看 `task_object_terminal_state` 中每个任务 qpos 的初值、终值和相对原记录
   终态的差异；
8. 最后用 viewer/replay 确认物体接触和语义任务结果。

`aligned_tracking.csv` 每行对应同一个 `reference_frame_index/policy_seq` 的
第一次出现，并同时保留两套各自的 logger index、时间、target、实测 q 和误差，
适合自己画图或进一步统计。`summary.json` 还保留
`ordinal_target_max_abs_difference_rad`，用于暴露未校正前因重复/漏帧造成的
ordinal target 差异。

## 5. replay 两次结果

replay binary 仍依赖你单独准备的 MuJoCo 3.2 动态库环境，不使用
`.venv_sim` 中的 MuJoCo 3.10：

```bash
./sample_data/ztj/replay_mujoco_csv \
  change_ckpt_track/data/20260612_144117_g1_sim_regular

./sample_data/ztj/replay_mujoco_csv \
  change_ckpt_track/data/20260612_144117_g1_sim_low_latency
```

replay 只按新的 `data.csv` qpos 展示仿真结果；它适合逐段查看机器人和任务
物体运动，但不会重新运行 checkpoint。

### 同时显示 rollout 与原 recording qpos

`change_ckpt_track/replay_mujoco_compare.py` 同时兼容 `change_ckpt` 和
`change_ckpt_track` 的结果目录。它会在正常的 rollout 和任务场景上额外绘制
一台**只参与渲染、不参与碰撞或动力学**的半透明 G1：

- 原材质、不透明的 G1：本次 regular/low-latency rollout 的实际 qpos；
- 青色半透明部分：原 recording 同一 source-clock 时刻的身体实际 qpos；
- 橙色半透明手指：原 recording 同一帧的实际手指 qpos。使用不同颜色是因为
  手指不属于 29-DOF body encoder reference，实际控制走独立 hand target 链路；
- 任务物体只显示本次 rollout 的一份，不会生成 ghost，也不会改变 replay 结果。

这个新 viewer 使用 `.venv_replay` 中的 MuJoCo 3.2，原 C++ replay 程序和源码
保持不变。直接传新结果目录即可；原 recording 路径会从 `run_metadata.json`
或 `launch_manifest.json` 自动找到，显式 50 Hz 模式还可从
`prepared_reference.npz` 的 metadata 获取：

```bash
# change_ckpt 旧方案
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt/data/20260720_144342_g1_sim_regular

# change_ckpt_track qpos-track 方案
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/20260720_144342_g1_sim_regular

.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/20260720_144342_g1_sim_low_latency
```

如果 sidecar 中的旧路径已失效，viewer 会按 recording 目录名在
`sample_data/**` 下查找；只有唯一匹配时才自动采用。存在多个同名目录时必须
显式指定：

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/20260720_144342_g1_sim_regular \
  --reference sample_data/ztj/20260612/20260720_144342_g1_sim
```

默认 `--ghost-mode source` 按本次 simulator 的 source clock 查看原 recording
的实际 qpos。当前 200 Hz simulator 每 tick 前进两个 400 Hz source row，因此
ghost 是原 recording 的隔行 200 Hz 采样；源数据结束后与 simulator 一样保持
最后一行。

如果要看“每个 policy 边界采样一次、按 50 Hz 保持”的构造轨迹，可显式运行：

```bash
# change_ckpt：由 legacy policy_seq/control_time_s 恢复边界行
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt/data/20260720_144342_g1_sim_regular \
  --ghost-mode reference

# change_ckpt_track：直接使用保存的 source_row_index
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/20260720_144342_g1_sim_regular \
  --ghost-mode reference
```

对 qpos-track NPZ，这些 policy 边界行直接来自保存的 `source_row_index`。对
旧 `change_ckpt` NPZ，viewer 用其中的 `policy_seq/control_time_s` 在原
recording 中恢复相应边界行。两者显示的都是**原 recording 的完整 qpos 在
policy 边界的 50 Hz sample-and-hold**；旧 `change_ckpt` 的 1024 维
`reference_motion` 缺少完整 root/task qpos，viewer 不会把它误当成可直接反解的
MuJoCo qpos。

默认 ghost 使用原 recording 的 pelvis 世界位置和朝向，能看到全局位置、姿态及
关节误差。若全局漂移使两个模型分得太远，希望只突出关节姿态差异，可把 ghost
固定到本次 rollout 的 pelvis：

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/20260720_144342_g1_sim_regular \
  --root-mode actual
```

常用控制：

- `Space`：暂停/继续；
- `G`：显示/隐藏 ghost；
- `R`：从所选区间开头重新播放；
- 暂停后 `Left` / `Right`：逐个 200 Hz sample 检查；
- `--ghost-alpha 0.2`：调整透明度；
- `--ghost-offset 0 0.5 0`：只为观察而把 ghost 平移，不改变数据；
- `--start-time 3 --end-time 6`：只看指定时间段；
- `--hide-ghost-hands`：只看 body reference；
- `--dry-run`：只检查 CSV 对齐、模型和 ghost geoms，不打开 viewer。

两个语义边界需要保留：

1. protocol v1 没有把 `root_pos` 发给 encoder；默认绘制原 pelvis xyz 是为了
   world-space 比较，不能解释成 checkpoint 收到了 root xyz target。
2. simulator 和 C++ deploy/reference publisher 是独立 wall-clock 线程。现有
   replay CSV 没有记录“这个 200 Hz sim tick 实际消费的 reference frame ID”。
   默认 `source` ghost 使用 simulator 保存的 source-row 时钟，并通过每行
   `policy_seq` 校验；可选的 `reference` ghost 使用设计时钟的 50 Hz 对齐。
   后者在正常运行中是正确的 nominal 映射，但若某次 publisher 发生
   duplicate/skip，不能仅凭 replay CSV 证明每个 sim tick 实际消费了哪一帧。
   严格审计仍应结合 `compare_qpos_track.py` 对
   `target_motion.csv` 的 frame-value 匹配结果；以后可在采集时显式保存
   `reference_frame_index/source_row_index` 消除这一限制。

## 6. 自检

比较器和 reference 构造器的纯 Python 测试：

```bash
.venv_sim/bin/python -m unittest -v \
  change_ckpt_track/test_qpos_reference_data.py \
  change_ckpt_track/test_qpos_track_publisher.py \
  change_ckpt_track/test_recorded_slot_lags.py \
  change_ckpt_track/test_launch_checkpoint_rollout.py \
  change_ckpt_track/test_run_task_sim_loop.py \
  change_ckpt_track/test_compare_qpos_track.py
```

ghost viewer 的无界面检查：

```bash
.venv_replay/bin/python -m unittest -v \
  change_ckpt_track/test_replay_mujoco_compare.py

.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/20260720_144342_g1_sim_regular \
  --dry-run
```
