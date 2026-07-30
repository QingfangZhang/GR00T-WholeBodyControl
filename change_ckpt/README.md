# 用记录数据对比 regular 与 low-latency checkpoint

这套工具用于把已有任务记录作为 reference，在 MuJoCo 中实时、闭环地比较 SONIC regular checkpoint 和 low-latency checkpoint。它不会修改 `gear_sonic/`、`gear_sonic_deploy/` 等原有源码；新增和修改的 Python 代码、配置、模型副本都位于 `change_ckpt/`，实验结果默认写入 `change_ckpt/data/`。运行时只读调用项目已有的 C++ deploy 可执行文件。

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

regular 和 low-latency encoder 的 reference 布局不同：regular 保留记录中 640 维、10 个 slot 的原始布局语义；low-latency 使用按 `policy_seq` 去重后的连续 50 Hz reference 帧。两种布局最终都通过 protocol v1 发送关节位置、关节速度、绝对 anchor orientation 和手目标，由 C++ deploy 在本地构造 encoder 输入。publisher 明确不发送 CSV 中原有的 `token_state`，从而避免绕过待测 encoder。

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
  --validate-recording sample_data/ztj/20260720_144342_g1_sim \
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
  --recording sample_data/ztj/20260720_144342_g1_sim \
  --checkpoint sonic_release/last.pt \
  --output change_ckpt/data/regular_encoder_token_validation.json
```

## 2. 运行前检查

统一 launcher 本身建议从 `.venv_sim` 运行；它会用同一个环境启动复制后的 MuJoCo simulator 和 CSV publisher，并启动现有 C++ deploy。先确保没有旧的 `run_sim_loop`、deploy 或 publisher 占用 DDS/ZMQ 端口，然后分别检查两套模型。

```bash
source .venv_sim/bin/activate

python change_ckpt/launch_checkpoint_rollout.py preflight \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260720_144342_g1_sim

python change_ckpt/launch_checkpoint_rollout.py preflight \
  --checkpoint regular \
  --recording sample_data/ztj/20260720_144342_g1_sim
```

为防止 TensorRT 在 GPU/hash 变化时重写原目录，low-latency 和 planner 的 ONNX、配置及现有 TRT cache 都已复制到 `change_ckpt/models/`。low-latency 默认读取：

- `change_ckpt/models/low_latency/model_encoder.onnx`
- `change_ckpt/models/low_latency/model_decoder.onnx`
- `change_ckpt/models/low_latency/observation_config.yaml`

regular 默认读取上一节导出的 `change_ckpt/models/regular/model_{encoder,decoder}.onnx` 和 `change_ckpt/observation_config_sonic_release.yaml`。若模型放在其他位置，可给 `preflight` 和 `run` 同时传入 `--encoder`、`--decoder`，必要时再传 `--obs-config`。

preflight 会检查模型维度、observation 配置、记录及场景文件、publisher 协议、C++ deploy 和动态库，并以正式参数实际加载一次 simulator 场景/资产、构造一次 publisher reference；它不会运行策略或验证任务效果。

## 3. 实时运行并默认保存 CSV

一次只运行一套。下面两个命令都会在当前终端统一启动 simulator、C++ deploy 和 publisher；默认打开 MuJoCo viewer、按实时节奏运行、到 source 末尾停止，并保存结果。

low-latency：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint low_latency \
  --recording sample_data/ztj/20260720_144342_g1_sim
```

regular：

```bash
python change_ckpt/launch_checkpoint_rollout.py run \
  --checkpoint regular \
  --recording sample_data/ztj/20260720_144342_g1_sim
```

输出目录分别形如：

```text
change_ckpt/data/20260720_144342_g1_sim_low_latency/
change_ckpt/data/20260720_144342_g1_sim_regular/
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
- `deploy_csv/action.csv`：C++ deploy 中本地 decoder 实际生成的 29 维 policy action。日志存在一帧历史对齐：`token_state[i]` 对应 `action[i+1]`，`action[0]` 是启动时的零 action 历史。与 `token_state.csv` 一起分析策略输出，不要改用 `data.csv` 中被清零的同名/相近字段。
- `target_motion.csv`：C++ deploy 合并后实际消费的 reference motion，用于核对 publisher 与 deploy 的 reference 是否对齐；它不是机器人实际 qpos 轨迹。
- `run_metadata.json`：无论是否保存 replay CSV都会写出停止原因、是否跌倒/出现非有限状态、实时因子、最大调度延迟、各阶段耗时、手状态发布完整性和任务物体终态。
- `launch_manifest.json`：checkpoint 类型、模型/配置的绝对路径与 SHA-256、初始化方式、时序和三条实际启动命令。
- `reference_diagnostics.json`、`prepared_reference.npz`：reference 布局、anchor orientation 恢复、手目标和准备好的 50 Hz stream 诊断数据。
- `sim.log`、`deploy.log`、`publisher.log`：三个进程的完整日志。

C++ decoder 的 10 帧状态历史在启动时由 0 填充，而不是从源 CSV 伪造。50 Hz 下填满需要 0.2 秒，所以定量对照应排除每次运行最初 0.2 秒；`launch_manifest.json` 也记录了这个 warmup。第一控制 tick 还会执行 C++ 自身的 heading 初始化，默认在 publisher tick 1 恢复记录的初始相对 yaw，这两项设置在 regular/low-latency 对照中应保持一致。

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
- 都排除最初 0.2 秒 decoder-history warmup。

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

## 参数帮助

脚本仍在迭代时，以本地源码和 `--help` 输出为准：

```bash
python change_ckpt/launch_checkpoint_rollout.py preflight --help
python change_ckpt/launch_checkpoint_rollout.py run --help
python change_ckpt/export_regular_g1_onnx.py --help
```
