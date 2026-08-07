# SONIC v1.1 qpos-track 实验记录

实验日期：2026-08-07。

本文件记录把 qpos-track 的 controller 换成 NVIDIA SONIC v1.1 后的模型校验、
运行条件和第一轮结果。这里的 reference 仍是原 recording 中机器人实际走过的
50 Hz qpos/qvel/pelvis quaternion，不使用原 CSV 的 `reference_motion` 或
`token_state`。

## 模型校验

模型保留在 `change_ckpt/models/v1.1/`，没有复制或移动。首次部署生成的 TensorRT
cache 也位于该目录。

| 文件 | SHA-256 |
|---|---|
| `model_encoder.onnx` | `fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae` |
| `model_decoder.onnx` | `34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509` |
| `observation_config.yaml` | `4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0` |

encoder/decoder 的 ONNX SHA-256 与 NVIDIA 官方 Git LFS object ID 一致；
`observation_config.yaml` 与仓库中的
`gear_sonic_deploy/policy/sonic_v1_1/observation_config.yaml` 字节完全相同。

- encoder：`[1,1751] -> [1,64]`；
- decoder：`[1,994] -> [1,29]`；
- G1 reference 窗口：canonical `[0,5,10,...,45]`；
- v1.1 使用 `motion_anchor_orientation_heading_10frame_step5`；wire 仍发送
  recording 的绝对世界系四元数，heading 相对旋转由 C++ observation registry
  计算；
- v1.1 不允许 recorded-lag 兼容模式。

## 运行条件

- reference：原 recording 的 50 Hz qpos-track；
- policy/reference：50 Hz；
- MuJoCo control：200 Hz；
- MuJoCo physics：1000 Hz；
- 无 viewer，保存 `data.csv`；
- future window：canonical；
- `root-assist none` 是无辅助结果；`xy/xyz` 是每个 control tick 后硬对齐 source
  root 的 oracle 诊断，不能视为 checkpoint 自身定位能力。

新结果根目录：

```text
change_ckpt_track/data/sonic_v1_1_comparison/
```

表中 `20260612_144127` 的 regular/low-latency none、xy、xyz 基线复用了此前已
通过实时质量门的 `change_ckpt_track/data/20260612_144127_g1_sim_*` 结果；没有
为了目录整齐而复制这些大文件。其余表中结果位于上述新根目录。

三个 recording 的 v1.1 none/xy 均完整跑到 source 结尾、未跌倒、无非有限状态，
并通过实时质量门。考虑到 `20260612_144127` 之前只有 xyz assist 才能把脚带到
正确高度，额外运行了该 recording 的 v1.1 xyz 条件，也通过质量门。

## 关节与 root tracking

`body RMSE` 是 29-DoF 实测关节相对本次 controller 实际消费的
`target_motion.csv` 的误差。root 误差则相对原 recording 同一 source-row；assist
覆盖的坐标为零是外部强制结果。

| recording | checkpoint | assist | body RMSE (rad) | root XY RMS (m) | root XYZ RMS (m) | yaw RMS (deg) |
|---|---|---:|---:|---:|---:|---:|
| `20260720_144342` | regular | none | 0.05973 | 0.16535 | 0.16555 | 1.618 |
|  | SONIC v1.1 | none | 0.10954 | 0.10443 | 0.10469 | 2.815 |
|  | low-latency | none | 0.13166 | 0.13569 | 0.13607 | 3.760 |
|  | regular | xy | 0.05791 | 0.00000 | 0.00751 | 1.668 |
|  | SONIC v1.1 | xy | 0.10616 | 0.00000 | 0.00707 | 2.956 |
|  | low-latency | xy | 0.12503 | 0.00000 | 0.00893 | 2.951 |
| `20260612_144127` | regular | none | 0.05558 | 0.05578 | 0.05637 | 0.996 |
|  | SONIC v1.1 | none | 0.06935 | 0.05448 | 0.05506 | 1.646 |
|  | low-latency | none | 0.07349 | 0.16112 | 0.16149 | 1.354 |
|  | regular | xy | 0.05168 | 0.00000 | 0.00802 | 1.312 |
|  | SONIC v1.1 | xy | 0.06480 | 0.00000 | 0.00616 | 1.907 |
|  | low-latency | xy | 0.07563 | 0.00000 | 0.01207 | 1.452 |
|  | regular | xyz | 0.05009 | 0.00000 | 0.00000 | 1.594 |
|  | SONIC v1.1 | xyz | 0.06982 | 0.00000 | 0.00000 | 2.325 |
|  | low-latency | xyz | 0.07472 | 0.00000 | 0.00000 | 3.528 |
| `20260722_154958` | regular | none | 0.05179 | 0.02009 | 0.02050 | 1.062 |
|  | SONIC v1.1 | none | 0.09845 | 0.04778 | 0.04789 | 1.098 |
|  | low-latency | none | 0.13595 | 0.09829 | 0.09840 | 1.463 |
|  | regular | xy | 0.04988 | 0.00000 | 0.00353 | 1.274 |
|  | SONIC v1.1 | xy | 0.09772 | 0.00000 | 0.00319 | 0.995 |
|  | low-latency | xy | 0.13282 | 0.00000 | 0.00408 | 1.507 |

在这些 recording 和同一 qpos reference 下，SONIC v1.1 的 29-DoF RMSE 在每个
none/xy/xyz 对照中都处于 regular 与 low-latency 之间。它说明 v1.1 能稳定闭环
跟踪这三段动作，但不能据此推断任务接触成功。

## 任务物体的客观终态

### `20260720_144342_g1_sim`

原 recording 的任务是拿起 scanner、放入抽屉并合上抽屉。

- v1.1 none：scanner 只移动约 `0.006 m`，最终位置距原 recording 终点约
  `0.806 m`；
- v1.1 xy：scanner 只移动约 `0.042 m`，最终位置距原 recording 终点约
  `0.809 m`；
- 两项的 lower drawer 终态都接近 recording 的关闭位置。

因此可以客观判断 v1.1 的 scanner 没有到达原 recording 的放置终态；抽屉终态
接近并不能补足拿取/放置阶段。最终仍应 replay 确认具体在哪个接触阶段失败。

### `20260612_144127_g1_sim`

正式任务是踩上垃圾桶后保持稳定。垃圾桶在该场景中的任务 qpos 不能记录脚与桶
之间的接触和“是否稳定站住”，所以 none/xy/xyz 的终态 qpos 都不能自动给出成功
标签。必须 replay v1.1 xyz，并与已验证过的 regular xyz 画面直接比较。

### `20260722_154958_g1_sim`

本轮没有为这个 recording 猜测语义任务标签，只报告 `can_base0` 的物理状态。
v1.1 none/xy 的罐体终态均与原 recording 终点有明显差异；是否仍满足原任务目标
需要根据该 recording 的真实任务定义和 replay 判断。

## 查看结果

带原 recording 同时刻 ghost 的推荐命令：

```bash
# scanner 任务，无辅助
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/sonic_v1_1_comparison/20260720_144342_g1_sim_sonic_v1_1

# scanner 任务，XY oracle 诊断
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/sonic_v1_1_comparison/20260720_144342_g1_sim_sonic_v1_1_root_assist_xy

# 踩垃圾桶任务，XYZ oracle 诊断
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/sonic_v1_1_comparison/20260612_144127_g1_sim_sonic_v1_1_root_assist_xyz

# 第三条 recording，XY oracle 诊断
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt_track/data/sonic_v1_1_comparison/20260722_154958_g1_sim_sonic_v1_1_root_assist_xy
```

汇总新根目录中的 rollout：

```bash
.venv_sim/bin/python change_ckpt_track/summarize_checkpoint_rollouts.py \
  --scan-root change_ckpt_track/data/sonic_v1_1_comparison
```

这条扫描命令不会自动加入上面说明的旧 `20260612_144127` 基线；完整对照数值已
固定记录在本文件的表格中，也可以把那些旧结果目录作为位置参数显式传给汇总
工具。

如果要做正式成功率统计，应先为每种任务定义接触、物体位姿和稳定时间阈值，再做
多次重复运行；当前结果是每个条件一次的确定性诊断，不是成功率估计。
