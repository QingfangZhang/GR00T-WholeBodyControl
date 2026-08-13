# SONIC v1.1 reference-motion 实验记录

本文件记录把 SONIC v1.1 接入 `change_ckpt` 后的验证和三份 headless
rollout。这里的身体目标来自原 `data.csv` 的 `reference_motion`，不是从原
机器人实际 `qpos` 构造的 qpos-track。

## 实现语义

- `reference_motion[0:580]` 的十槽 29-DoF q/dq 原样保留。
- 旧 regular 的最后 60 维 6D orientation 先结合 recording pelvis quaternion
  恢复为十槽世界系四元数。
- publisher 通过 protocol v1 发送 q、dq 和世界系四元数，不发送旧
  `token_state`。
- C++ 根据 v1.1 rollout 当前的 robot heading，在运行时构造
  `motion_anchor_orientation_heading_10frame_step5`。
- 模型为 `change_ckpt/models/v1.1/model_encoder.onnx`（1751→64）和
  `model_decoder.onnx`（994→29）。六个完整日志条件中的 token/action 均为
  finite，维度分别为 64/29。

通过相邻 `policy_seq` 的 reference slot 重叠关系推断，三份 recording 的
代表性十槽 policy lag 约为：

| Recording | recorded slot lags |
|---|---|
| `20260720_144342` | `[0,5,9,9,9,9,9,9,9,9]` |
| `20260612_144127` | `[0,4,4,4,4,4,4,4,4,4]` |
| `20260722_154958` | `[0,5,8,8,8,8,8,8,8,8]` |

这些 lag 不是 CSV 中显式保存的字段，而是比较相邻 policy 帧数值后得到的；
本实验保留每个 source row 的原始十槽数值，没有把它们重新采样成 canonical
`[0,5,10,...,45]`。

## 运行结果

所有结果都使用 200 Hz MuJoCo control、1 kHz physics、50 Hz reference、
headless 和保存 replay CSV。

| Recording | Root assist | 严格时序有效 | RTF | 最大 lag | 稳定性/任务物体指标 |
|---|---|---:|---:|---:|---|
| `20260720_144342` | none | 是 | 0.999981 | 9.55 ms | 未跌倒；抽屉没有运动；scanner 位移只有 source 的 12.7%，未复现任务 |
| `20260720_144342` | xy | 否 | 0.979933 | 255.05 ms | 未跌倒；抽屉完成 source 的 101.8%，scanner 位移为 98.7%，终点位置误差 3.61 cm；RTF、最大 lag、wall-sim drift 均未过门限并发生 1 次 rebase，不能作为正式实时结果 |
| `20260612_144127` | none | 是 | 0.999965 | 5.07 ms | 未跌倒；踏板/桶盖终值相对 source 的误差分别为 0.00059/0.00340 rad |
| `20260612_144127` | xy | 是 | 0.999873 | 34.75 ms | 未跌倒；踏板/桶盖终值误差分别为 0.00028/0.00160 rad |
| `20260722_154958` | none | 是 | 0.999978 | 14.74 ms | 未跌倒；物体终点位置误差 2.18 cm，姿态误差 28.1° |
| `20260722_154958` | xy | 是 | 0.999972 | 10.11 ms | 未跌倒；物体终点位置误差 1.12 cm，姿态误差 31.2° |

这里的物体状态只是物理筛查。`20260612_144127` 的正式任务是踩上垃圾桶后
保持稳定；`20260720_144342` 的正式任务是拿起物体、放入抽屉并合上抽屉，
两者都仍需要 replay 目视确认完整接触过程。`20260722_154958` 的语义任务没有
在本文件中猜测，只报告物体终态。尤其是 `20260720_144342` 的 XY 条件虽然
物体终态很接近原记录，但严格实时门没有通过，不能据此宣称 v1.1 已正式完成
任务。

复杂抽屉场景还做了 `--no-deploy-csv-logs` 时序隔离：none 条件有效；XY
条件 RTF 达到约 1.0，但仍出现大于 50 ms 的单次调度峰值，而且任务终态随
运行发生变化。这说明问题不只是 deploy CSV 写盘，不能用轻量日志结果替代
标准完整日志结果。

## 输出目录

标准完整日志结果：

```text
change_ckpt/data/20260720_144342_g1_sim_sonic_v1_1/
change_ckpt/data/20260720_144342_g1_sim_sonic_v1_1_root_assist_xy/
change_ckpt/data/20260612_144127_g1_sim_sonic_v1_1/
change_ckpt/data/20260612_144127_g1_sim_sonic_v1_1_root_assist_xy/
change_ckpt/data/20260722_154958_g1_sim_sonic_v1_1/
change_ckpt/data/20260722_154958_g1_sim_sonic_v1_1_root_assist_xy/
```

使用带原 recording ghost 的比较播放器：

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt/data/20260720_144342_g1_sim_sonic_v1_1

.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt/data/20260720_144342_g1_sim_sonic_v1_1_root_assist_xy
```

把目录替换为另外四个结果即可查看其他条件。
