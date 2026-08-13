# 用记录数据对比 regular、low-latency 与 SONIC v1.1 checkpoint

这套工具用于把已有任务记录作为 reference，在 MuJoCo 中实时、闭环地比较
SONIC regular、low-latency 和 SONIC v1.1 checkpoint。它不会修改
`gear_sonic/`、`gear_sonic_deploy/` 等原有源码；新增和修改的 Python 代码、
配置、模型副本都位于 `change_ckpt/`，实验结果默认写入 `change_ckpt/data/`。
运行时只读调用项目已有的 C++ deploy 可执行文件。

这里做的不是“用新 action 离线覆盖原 CSV 的 qpos”。数据流是：

```text
记录 data.csv 中的 reference/手目标
  -> CSV publisher（ZMQ protocol v1，不发送 token_state）
  -> C++ deploy 本地 encoder + 本地 decoder
  -> DDS 关节命令
  -> MuJoCo 闭环动力学
  -> 新的仿真 data.csv
```

因此 encoder token、decoder action、机器人状态历史、PD 控制和物理反馈都来自本次运行。默认的 `--root-assist none` 下，源 CSV 的 qpos/qvel 只用于初始化机器人和任务物体并提供 reference 时间轴，不会逐帧强制写回机器人状态。只有显式启用下文的 root-assist 诊断选项时，才会逐步补偿选中的 floating-base 分量。

## 仿真时序

| 环节 | 200 Hz 默认模式 | 400 Hz 诊断模式 |
|---|---:|---:|
| 源 CSV logger | 400 Hz (`2.5 ms`) | 400 Hz (`2.5 ms`) |
| MuJoCo 控制/DDS 循环 | 200 Hz (`5 ms`) | 400 Hz (`2.5 ms`) |
| MuJoCo 物理积分 | 1 kHz (`1 ms`) | 2 kHz (`0.5 ms`) |
| 每个控制周期的物理 substep | 5 | 5 |
| 每个控制周期消费的源 CSV 行 | 2 | 1 |
| reference 发布及策略更新 | 50 Hz | 50 Hz |
| viewer 刷新 | 50 Hz | 50 Hz（正式测试仍应关闭） |

默认仍是 `--sim-frequency 200`。实验性的 400 Hz 模式使用
`--sim-frequency 400`；launcher 会自动设置 `control_dt=0.0025` 和
`physics_dt=0.0005`，source 和策略时钟则分别保持 400 Hz、50 Hz。因此
每个 50 Hz policy action 大约保持 8 个仿真控制 tick。

400 Hz 会提高 source root 对齐、DDS 状态反馈、PD 重算和接触积分的时间
分辨率，但它不是把同一个 checkpoint 输出简单“插值成更精确的轨迹”：闭环
离散化也随之改变，而且每秒 MuJoCo 步数和 DDS/CSV tick 数都翻倍。只有无界面
运行的实时性校验通过后，才可把它作为 200 Hz/400 Hz 对照实验；原 CSV 的
logger 是 400 Hz 这一事实本身不能证明采集系统的控制环也是 400 Hz。

本机对 `20260612_144127_g1_sim` 的实测也说明了这一区别：400 Hz、保存 CSV、
无 root assist 的 regular 运行可以通过实时校验；同样配置再加逐 tick
`--root-assist xyz` 时则无法维持实时，因为每次 hard alignment 后额外执行的
`mj_forward` 已把平均周期成本推到 2.5 ms 预算附近。不要把后一条掉速轨迹用于
checkpoint 精度结论。

每次运行都必须检查 `run_metadata.json`：要求 `wall_clock_timing_valid=true`、`real_time_factor` 接近 1、没有 deadline rebase，并且手状态没有 overrun。程序在跌倒、非有限状态或时序无效时返回非零；不能仅凭参数假定实验有效。

regular、SONIC v1.1 和 low-latency encoder 的 reference 布局不同：regular
与 SONIC v1.1 都保留记录中 640 维、10 个 slot 的原始布局语义；low-latency
使用按 `policy_seq` 去重后的连续 50 Hz reference 帧。三种布局最终都通过
protocol v1 发送关节位置、关节速度、绝对 anchor orientation 和手目标，由
C++ deploy 在本地构造 encoder 输入。publisher 明确不发送 CSV 中原有的
`token_state`，从而避免绕过待测 encoder。

SONIC v1.1 不会直接复用旧 640 维中的最后 60 维。Python 先用 recording
pelvis quaternion 把 regular 的相对 6D orientation 恢复为十个 slot 的世界系
reference quaternion；C++ 再结合 v1.1 rollout 当前的 robot heading，实时构造
`motion_anchor_orientation_heading_10frame_step5`。因此前 580 维 q/dq 与十个
recorded slots 保持不变，heading normalization 使用 v1.1 的新定义。若原数据的
future slots 已被截成 `[0,5,9,9,...]` 等形式，本实验会原样保留；它测试的是
“同一个实际采集 reference 信号换 controller”，不是 canonical
`[0,5,10,...,45]` 重建实验。

## 1. 导出并验证 regular ONNX

regular 原 checkpoint 是 `sonic_release/last.pt`，需要先在 **isaaclab 环境**中导出。导出脚本不启动 Isaac Sim，它直接从 checkpoint 重建固定 G1 模式的 encoder 和 dynamic decoder：

- `model_encoder.onnx`：`1751 -> 64`
- `model_decoder.onnx`：`994 -> 29`

在项目根目录运行：

```bash
conda activate isaaclab
python change_ckpt/export_regular_g1_onnx.py \
  --checkpoint sonic_release/last.pt \
  --output-dir change_ckpt/models/regular \
  --validate-recording sample_data/ztj/20260612/20260720_144342_g1_sim \
  --validate-frames 575
```

`--validate-recording` 会对每个唯一 `policy_seq` 取一帧，同时比较：

- checkpoint 重建出的 PyTorch encoder 与 CSV `token_state[0:64]`；
- 导出的 ONNX encoder 与 CSV token；
- ONNX encoder 与 PyTorch encoder。

当前这份记录共有 575 个唯一 policy frame，已验证 575/575 的三组比较均为 `max_abs = 0`。这证明 regular G1 encoder 的历史 reshape、MLP、FSQ 以及 CSV 行对齐被精确复现；它不证明换 checkpoint 后闭环任务必然成功。导出的两个模型默认位于 `change_ckpt/models/regular/`，不会覆盖原 checkpoint。

如果只想独立复查 CSV 的 640 维输入和原 checkpoint token，也可在 isaaclab 环境运行：

```bash
python change_ckpt/validate_regular_encoder_tokens.py \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim \
  --checkpoint sonic_release/last.pt \
  --output change_ckpt/data/regular_encoder_token_validation.json
```

## 2. 运行前检查

统一 launcher 本身建议从 `.venv_sim` 运行；它会用同一个环境启动复制后的 MuJoCo simulator 和 CSV publisher，并启动现有 C++ deploy。先确保没有旧的 `run_sim_loop`、deploy 或 publisher 占用 DDS/ZMQ 端口，然后分别检查两套模型。

source-history prefill 现在是默认启动方式。第一次运行或部署源码变化后，先构建
隔离的 wrapper（不会修改或替换官方 deploy binary）：

```bash
source .venv_sim/bin/activate

.venv_sim/bin/python change_ckpt/build_source_history_deploy.py

python change_ckpt/launch_checkpoint_rollout.py preflight \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim

python change_ckpt/launch_checkpoint_rollout.py preflight \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim

python change_ckpt/launch_checkpoint_rollout.py preflight \
  --checkpoint sonic_v1_1 \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim
```

为防止 TensorRT 在 GPU/hash 变化时重写原目录，low-latency 和 planner 的 ONNX、配置及现有 TRT cache 都已复制到 `change_ckpt/models/`。low-latency 默认读取：

- `change_ckpt/models/low_latency/model_encoder.onnx`
- `change_ckpt/models/low_latency/model_decoder.onnx`
- `change_ckpt/models/low_latency/observation_config.yaml`

regular 默认读取上一节导出的 `change_ckpt/models/regular/model_{encoder,decoder}.onnx` 和 `change_ckpt/observation_config_sonic_release.yaml`。若模型放在其他位置，可给 `preflight` 和 `run` 同时传入 `--encoder`、`--decoder`，必要时再传 `--obs-config`。

SONIC v1.1 默认读取：

- `change_ckpt/models/v1.1/model_encoder.onnx`
- `change_ckpt/models/v1.1/model_decoder.onnx`
- `change_ckpt/models/v1.1/observation_config.yaml`

preflight 会检查模型维度、observation 配置、记录及场景文件、publisher 协议、C++ deploy 和动态库，并以正式参数实际加载一次 simulator 场景/资产、构造一次 publisher reference；它不会运行策略或验证任务效果。

## 3. 实时运行并默认保存 CSV

一次只运行一套。下面两个命令都会在当前终端统一启动 simulator、C++ deploy 和 publisher；默认打开 MuJoCo viewer、按实时节奏运行、到 source 末尾停止，并保存结果。

两套 launcher 统一使用同一种 offset 语义：原 CSV 的**第二个**连续
`policy_seq` group 固定记为 `--start-policy-offset 0`，无论第一个 group 是否
完整。默认值是 `10`，因此默认接管点是原 CSV 的第 12 个 group（raw group
offset 11）。用户 offset、实际 raw offset 和 `policy_seq` 都会写入
`launch_manifest.json`。

low-latency：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim
```

regular：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim
```

SONIC v1.1：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint sonic_v1_1 \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim
```

输出目录分别形如：

```text
change_ckpt/data/20260720_144342_g1_sim_low_latency/
change_ckpt/data/20260720_144342_g1_sim_regular/
change_ckpt/data/20260720_144342_g1_sim_sonic_v1_1/
```

目录名固定为 `<原始记录文件夹名>_<checkpoint>`，不再加入运行时间。对同一记录和同一 checkpoint 重复运行时，launcher 会先删除整个旧同名目录，再写入本次结果，避免新旧 `data.csv`、日志或模型快照混在一起；不要把需要保留的手工文件放入该目录。

400 Hz 模式会额外加入 `_400hz`，不会覆盖同条件的 200 Hz 结果。例如：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260612_144127_g1_sim \
  --sim-frequency 400 \
  --root-assist xyz \
  --no-viewer
```

对应输出为
`change_ckpt/data/20260612_144127_g1_sim_regular_400hz_root_assist_xyz/`。
同一套 400 Hz 条件重复运行仍会覆盖自己的同名目录。

### 可选：运行时补偿 source root

为了判断“任务失败主要是不是 root 平移没有跟上”，可以先做一个 oracle/root-assisted 诊断实验。推荐先只补水平位置：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260720_144342_g1_sim \
  --root-assist xy
```

`--root-assist xy` 在每个 MuJoCo 控制/物理周期结束后，把机器人 floating base 的 `qpos[x,y]` 及 `qvel[vx,vy]` 对齐到当前源 recording 行；然后才写本轮 CSV，并让下一轮 policy 读取对齐后的状态。在 200 Hz 模式中每次跳过一行源数据，在 400 Hz 模式中逐行对齐。它不会改 `z`、root orientation、身体/手关节或任务物体 qpos。源数据结束并保持末帧时，对应速度会置零。

如果确实需要连高度一起补，可以使用 `--root-assist xyz`；默认 `--root-assist none` 与原有行为完全相同。辅助结果使用独立目录，避免覆盖无辅助基线：

```text
change_ckpt/data/20260720_144342_g1_sim_regular_root_assist_xy/
change_ckpt/data/20260720_144342_g1_sim_regular_root_assist_xyz/
```

同一 recording、checkpoint 和 assist 模式重复运行仍会覆盖它自己的同名目录。`run_metadata.json` 的 `root_assist` 字段会记录模式、补偿前/后的 root 误差和耗时。

这个选项会真实改变闭环仿真的状态、接触和下一步策略输入，所以应在运行时使用，而不是只在 replay 时平移画面；但它使用了原始成功轨迹的未来真值，结果只能回答“补上 root 后是否恢复任务”，不能算 checkpoint 独立完成任务。比较 regular 与 low-latency 时，两边必须使用相同的 root-assist 模式，并同时保留 `none` 基线。

按 `Ctrl-C` 可协调停止子进程。需要无界面运行时加 `--no-viewer`：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --no-viewer
```

viewer 同步也占用控制周期。如果带界面运行的 `run_metadata.json` 显示时序无效，应把这次轨迹视为无效实验，并用 `--no-viewer` 做正式定量对照；不要用一个有效 headless 结果替代另一个已掉速的 viewer 结果。

若要诊断内部 token/action CSV 写入是否造成实时瓶颈，可以显式添加
`--no-deploy-csv-logs`。它仍保存 MuJoCo replay `data.csv`，但不生成
`deploy_csv/` 和 `target_motion.csv`，因此不能用来分析内部 token/action；输出
目录会额外带 `_no_deploy_csv`，不会覆盖标准完整日志结果。该选项只用于时序
隔离，默认始终为完整 deploy 日志。

只想做无界面且不生成 replay `data.csv` 的冒烟测试，可使用：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --no-viewer \
  --no-save-csv
```

`--no-save-csv` 只关闭 simulator 的 replay `data.csv`。launcher 仍会在 `change_ckpt/data/` 写 metadata、manifest、进程日志、reference 诊断和 C++ `deploy_csv`，以便确认实际使用了哪个模型以及本地 encoder/decoder 是否运行；当前没有“完全不产生任何输出”的 unified-run 模式。

为避免磁盘刷新破坏高频实时性，运行期间只在内存缓存紧凑的 replay 数值，并把 C++ CSV 暂存到 `/tmp`；进程停止后才合成完整 1579 列 `data.csv`，并把所有文件复制回本次 `change_ckpt/data/<run>/`。最终输出位置不变。身体/IMU DDS 仍在控制线程同步发布；两路手状态使用所选仿真控制频率，只在 MuJoCo 求解释放 GIL 时并行序列化，overrun 会使实验无效。

## 4. 输出文件分别表示什么

默认运行目录中最重要的文件如下：

- `data.csv`：本次 MuJoCo 闭环仿真的状态轨迹，可供 replay。它的 qpos/qvel 是新仿真结果；源 `policy_seq/reference_motion` 列仅为时间轴与来源追踪，特别是 low-latency 运行时，它们不是新 encoder 的实际 1247 维输入。DDS 不提供内部 token/raw action，因此这里的 `policy_valid` 为 0，`token_state` 和 raw-action 列被清零，不能用它们分析新 checkpoint。
- `deploy_csv/token_state.csv`：C++ deploy 中本地 encoder 实际生成的 64 维 token。
- `deploy_csv/action.csv`：C++ deploy 中 decoder 使用/生成的 29 维 policy action。默认 source-history 模式下，index 0..8 是插入的旧 controller history，第一个 live state 是 index 9，第一个新 decoder action 是 index 10；不要再把 `action[0]` 解释成零启动 action。与 `token_state.csv` 一起分析策略输出，不要改用 `data.csv` 中被清零的同名/相近字段。
- `target_motion.csv`：C++ deploy 合并后实际消费的 reference motion，用于核对 publisher 与 deploy 的 reference 是否对齐；它不是机器人实际 qpos 轨迹。
- `run_metadata.json`：无论是否保存 replay CSV都会写出停止原因、是否跌倒/出现非有限状态、实时因子、最大调度延迟、各阶段耗时、手状态发布完整性和任务物体终态。
- `launch_manifest.json`：checkpoint 类型、模型/配置的绝对路径与 SHA-256、初始化方式、时序和三条实际启动命令。
- `reference_diagnostics.json`、`prepared_reference.npz`：reference 布局、anchor orientation 恢复、手目标和准备好的 50 Hz stream 诊断数据。
- `sim.log`、`deploy.log`、`publisher.log`：三个进程的完整日志。

C++ decoder 默认使用 source CSV 中接管点之前九个 50 Hz 实测状态预填历史，
第十个状态来自第一条 live MuJoCo DDS 状态，其 last action 从 source current
恢复，因此启动时没有 zero padding。新 controller 接管后的前十次推理中，
action/state history 仍逐帧包含旧 controller 的部分；从第十一次推理开始，十帧
history 才全部由新 controller 本轮闭环产生。第一控制 tick 仍会执行 C++ 自身
的 heading 初始化，默认在 publisher tick 1 恢复记录的初始相对 yaw。

### 默认启动方式与零历史诊断

默认运行使用独立的 source-history wrapper；官方 `gear_sonic_deploy` 源码和
原 release binary 不会被修改。构建命令是：

```bash
.venv_sim/bin/python change_ckpt/build_source_history_deploy.py
```

launcher 的公开 offset 从第二个 raw group 开始编号：公开 offset `10` 对应 raw
offset `11`，即原 CSV 第 12 个 group。普通命令无需再写
`--source-history-prefill`。若要做同初态、零历史的诊断，必须显式关闭默认 prefill：

```bash
# 严格对照 A：同一个 phase-matched source 初态、同一个隔离 wrapper，但不预填历史
.venv_sim/bin/python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260612_144154_g1_sim \
  --start-policy-offset 10 \
  --no-source-history-prefill \
  --source-state-init \
  --chunk-size 64 \
  --lookahead 64 \
  --no-viewer

# 默认 B：此前 9 个 source policy 状态预填历史
.venv_sim/bin/python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260612/20260612_144154_g1_sim \
  --start-policy-offset 10 \
  --chunk-size 64 \
  --lookahead 64 \
  --no-viewer
```

结果目录不再写入 `start_offset` 或 `source_history_prefill`：

```text
<recording>_regular/
```

因此同一 recording/checkpoint/root-assist 下，改 offset 或切换 history 模式会覆盖
同名目录；准确配置以该目录内的 `launch_manifest.json` 为准。需要同时保留严格
A/B 时，应在运行下一组前复制或重命名上一组目录。

`--source-state-init` 是专门用于因果对照的诊断选项：它恢复与 prefill 完全相同
的 phase-matched qpos/qvel 和 source clock，并调用与 prefill 相同的隔离 wrapper，
但不向 StateLogger 插入任何记录历史。这样两组的 controller 源码和 binary 也
完全相同。若省略该选项，普通 offset 起跑会直接使用 policy boundary 的 CSV 行，
不能与 phase-matched prefill 构成只差 history 的严格 A/B。

regular encoder 每次需要看到当前帧至 `current+45`。正式 A/B 将滚动 packet 从
默认 50 帧扩大到 64 帧，为异步 ZMQ input thread 留出 18 帧余量；否则在包安装
恰好落后时，官方 streamed-motion guard 可能把旧 reference 多保持一个 20 ms
tick。正式结果还要求两组实际保存的 `target_motion.csv` 逐元素相同；仅
`publisher late_ticks=0` 并不足以证明 controller 实际消费时序一致。

prefill 以每个 policy group 第一行保存的 `policy_received_dof_pos[0:43]` 为
原策略实际收到的 measured joint q，并在该 group 之前 20 ms 的 400 Hz
`qpos/qvel` 时间线上拟合其采样时刻；29 个身体关节跳过左右手、按 C++ 相同映射
转为 IsaacLab 顺序并减去 default angles。dq、pelvis quaternion 与 angular
velocity在拟合时刻插值，last action 取该 group 的
`policy_last_action_in[0:29]`。这份 recording 的拟合时刻平均比 policy metadata
首次写入 CSV 早约 10.7 ms，因此不能直接用每组第一条 CSV 行替代 decoder
状态。

只把前 9 个 source policy 状态塞入 logger；第 10 帧仍由第一次 live DDS
CONTROL tick 产生，同时用 current source 的 last-action 替换这一帧原本的零
action。MuJoCo 也从 current 的相位匹配状态启动；程序严格核对首次 live
q/dq/IMU 与 source current state，不对齐就中止。在 20260612_144154 上，离线
regular decoder 使用 source token 时，相位匹配状态把首个 raw-action RMSE 从
按 policy 首行构造的 `0.00598` 降至约 `0.00023`，所以最终实现使用相位匹配值。

`source_history_prefill.json` 和 `launch_manifest.json` 记录使用的 policy_seq、
CSV 行、源 CSV/模型/程序的 SHA-256 与字段公式。preflight 还会核对独立 deploy
与 wrapper、官方源码、关键 header 和构建配置的 hash；代码变化后必须重新运行
build 命令，不能静默沿用旧 binary。由于当前独立 wrapper 通过普通 StateLogger API
插入历史，开启 deploy CSV 时 `q/dq/action.csv` 的 index 0..8 是 prefill，
第一个 live token/state 是 index 9，第一个新 decoder action 出现在 index 10；
不要继续套用普通运行中 `action[0]` 为零的行号假设。

source-state-aligned 输出的 replay `data.csv` 中，复制的 `policy_seq` 和
`reference_motion` 跟随物理 source timeline。由于状态相位匹配，replay 首行与
publisher 的 policy boundary 可能不是同一个 CSV 行；这些复制列只用于
provenance 和 ghost 时间轴，不能视为新 controller 的真实 policy telemetry。
实际 publisher 时钟和两套初始时钟均记录在 `launch_manifest.json`。

该诊断只恢复 decoder 的五类历史输入；它不恢复 MuJoCo 的旧 ctrl、contact、
`qacc_warmstart`、求解器内部状态或原采集时的 DDS 控制相位。因此 prefill 能
消除启动瞬态，也不代表之后一定不会 root drift。20260612_144154 的实测结果
见 `change_ckpt/SOURCE_HISTORY_PREFILL_RESULTS.md`。

## 5. Replay

在已经配置好该 replay 可执行文件所需 MuJoCo 3.2 动态库的终端，从项目根目录运行：

```bash
./sample_data/ztj/replay_mujoco_csv \
  change_ckpt/data/<本次运行目录>/data.csv
```

这里必须传入具体的 `data.csv` 文件路径，不能只传 `change_ckpt/data/<本次运行目录>`。当前 replay 对目录的自动处理可能重新暂存 XML，却没有保留任务资源链接，进而再次出现 `pelvis.STL` 或 `task_assets` 找不到的问题。

## 6. 怎样做公平对照和判断效果

regular 与 low-latency 两次运行应保持以下条件相同：

- 同一个 `--recording`、`--start-policy-offset` 和场景快照；
- 相同的 `--sim-frequency` 及 control/source/physics/reference 时序；
- 相同的 `--base-sample`、heading correction、手目标和停止条件；
- 相同的 `--root-assist` 模式（正式无辅助基线应为 `none`）；
- 相同机器负载，且都确认 `real_time_factor` 接近 1；
- 都使用默认 source-history 初始化；新 controller 前十次推理属于旧/新 history
  逐步替换期，从第十一次推理起 action history 才完全来自新 controller。

对照中只切换 `--checkpoint` 及其匹配的 encoder、decoder、observation config。先核对两个 `launch_manifest.json`，再结合 viewer/replay、`run_metadata.json`、`target_motion.csv` 和 `deploy_csv` 判断：机器人是否跌倒或出现非有限状态、动作是否振荡、脚是否严重打滑、手和身体是否正确接触任务物体、任务物体最终状态是否达到原任务目标。

“preflight 通过”“token/action 均为有限值”或“机器人没有跌倒”都不等于任务成功。新 checkpoint 是在闭环动力学中重新产生动作，轨迹与原记录发生偏离是正常现象；最终结论必须依据实际任务完成情况，不能假定 low-latency checkpoint 一定能复现原 regular checkpoint 的成功结果。

完成一对运行后，可以生成同一套物理指标报告：

```bash
python change_ckpt/compare_checkpoint_rollouts.py \
  change_ckpt/data/<regular_run> \
  change_ckpt/data/<low_latency_run> \
  --output change_ckpt/data/regular_vs_low_latency.json
```

当前已经完成的一对有效 headless 运行如下；两次 manifest 均确认 encoder、decoder、配置和 planner 从 `change_ckpt/` 加载：

- regular：`20260722_234705_regular`，RTF 1.000，无跌倒/非有限状态；下层抽屉终态达到源运动的 103.8%，scanner 从初始位置移动 0.734 m（源记录 0.809 m）。
- low-latency：`20260722_234141_low_latency`，RTF 1.000，同样稳定；但下层抽屉终态只保留 53.9% 的源运动进度，scanner 仅移动 0.020 m。

所以自动物体状态筛查表明：这次 low-latency 运行虽然身体稳定、encoder/decoder 输出均有限，但没有复现 regular 的 scanner 操作。该判断是物理状态筛查，不是语义成功标注；仍应在实时 viewer 或 replay 中目视确认接触过程。完整报告位于 `change_ckpt/data/20260722_234705_regular_vs_234141_low_latency.json`。

SONIC v1.1 的 reference-motion 接入、三份 recording 的运行指标和 replay
命令记录在 `change_ckpt/SONIC_V1_1_REFERENCE_RESULTS.md`。

## 参数帮助

脚本仍在迭代时，以本地源码和 `--help` 输出为准：

```bash
python change_ckpt/launch_checkpoint_rollout.py preflight --help
python change_ckpt/launch_checkpoint_rollout.py run --help
python change_ckpt/export_regular_g1_onnx.py --help
```
