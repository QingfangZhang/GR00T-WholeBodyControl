# Teleopit 使用 SONIC reference_motion 的实验

## 实验问题

本实验不再把原控制器执行后的机器人实际 qpos 全部作为 Teleopit 的
目标，而是尽可能恢复当时送给 SONIC 的动作意图，再交给 Teleopit
tracker：

```text
SONIC reference_motion slot 0 的 29 维关节姿态
+ 恢复到世界系的 reference pelvis 四元数
+ 原 recording 同一 policy_seq 的实际 root xyz
→ 50 Hz Teleopit reference pose
→ Teleopit FK 和相邻帧差分
→ 167 维当前观测 + 在线十帧历史
→ track_g1.onnx 的 29 维动作
```

这不是纯 `reference_motion` 输入，因此模式名为
`sonic_reference_hybrid`。SONIC 的 640 维 `reference_motion` 没有
reference root xyz，而且其中保存的是 reference pelvis 相对实际机器人的
朝向，不是独立的世界系朝向。这里用原 recording 的实际 root 平移补足
xyz，并用记录的实际 pelvis 四元数乘相对朝向来恢复 reference 世界系
四元数。Teleopit 使用的 anchor 是 `torso_link`，所以代码再通过
Teleopit G1 模型的 FK 得到 torso 状态。手指仍由同一时刻的
`left_hand_q/right_hand_q` 独立驱动。

SONIC 的 slot 1--9 是未来目标，没有被误当作 Teleopit 历史。Teleopit
历史仍是运行时的当前完整观测加过去九个完整观测。两个 CSV 解析路径
按 `policy_seq` 对齐；三份测试记录的控制时间和手目标最大对齐误差均为
零。朝向恢复沿用 SONIC adapter 的 `previous-index5` 约定，并写入产物
metadata。

## 运行方法

以 `20260720_144342_g1_sim` 和 root xy assist 为例：

```bash
Teleopit_rollout/.venv/bin/python \
  Teleopit_rollout/launch_teleopit_rollout.py \
  sample_data/ztj/20260612/20260720_144342_g1_sim \
  --reference-source sonic_reference_hybrid \
  --root-assist xy --overwrite
```

其他 recording 只需替换输入目录。原 qpos-track 仍是默认模式；不写
`--reference-source` 时旧命令和旧输出命名不变。

自动输出名为：

```text
<recording>_teleopit_sonic_reference_hybrid_no_root_assist
<recording>_teleopit_sonic_reference_hybrid_root_assist_xy
<recording>_teleopit_sonic_reference_hybrid_root_assist_xyz
```

## 已完成的运行

以下七个运行都到达 reference 末尾，`fallen=false`、
`invalid_state=false`，并生成了完整性标记。所有时钟基于仿真步数；即使
实际运行速度低于实时，也不会让 reference 与仿真错位。

| Recording | Assist | 身体关节 RMSE | root xy 平均误差 | root z 平均误差 | root 朝向平均误差 | 任务状态摘要 |
|---|---|---:|---:|---:|---:|---|
| `20260720_144342` | none | 0.0971 rad | 0.0850 m | 0.00476 m | 6.77° | scanner 终点位置误差 0.783 m，位移仅为原始的 9.46% |
| `20260720_144342` | xy | 0.1024 rad | 0.00031 m | 0.00508 m | 7.29° | scanner 终点位置误差 0.0225 m，位移为原始的 101.47% |
| `20260612_144127` | none | 0.0951 rad | 0.0893 m | 0.00371 m | 6.88° | pedal/lid 没有运动 |
| `20260612_144127` | xy | 0.0997 rad | 0.00011 m | 0.00327 m | 6.45° | pedal/lid 没有运动 |
| `20260612_144127` | xyz | 0.1060 rad | 0.00011 m | 0.00002 m | 5.82° | pedal/lid 仍没有运动 |
| `20260722_154958` | none | 0.1093 rad | 0.0587 m | 0.00272 m | 3.82° | 物体终点位置误差 0.918 m，产生了非原始轨迹的较大运动 |
| `20260722_154958` | xy | 0.1125 rad | 0.00059 m | 0.00243 m | 3.85° | 物体终点位置误差 0.0187 m，位移为原始的 100.54% |

“root 朝向平均误差”是机器人相对其身体 reference 的误差，不是物体
朝向误差。任务是否语义成功也不能只由终点位置判断，仍需看 replay。

## 与原 Teleopit qpos-track 的对比

下面比较同一 Teleopit checkpoint、同一场景和同一 assist，只替换目标
构造方式。物体姿态误差取最终自由关节四元数的最短旋转角。

| Recording / assist | Reference | 身体关节 RMSE | 物体终点位置误差 | 物体终点姿态误差 | 物体位移/原始位移 |
|---|---|---:|---:|---:|---:|
| `20260720` none | actual qpos | 0.0748 rad | 0.797 m | 174.34° | 6.53% |
| `20260720` none | SONIC hybrid | 0.0971 rad | 0.783 m | 174.37° | 9.46% |
| `20260720` xy | actual qpos | 0.0774 rad | 0.0549 m | 91.28° | 100.20% |
| `20260720` xy | SONIC hybrid | 0.1024 rad | **0.0225 m** | **7.43°** | 101.47% |
| `20260722` none | actual qpos | 0.0727 rad | 0.249 m | 15.05° | 0.01% |
| `20260722` none | SONIC hybrid | 0.1093 rad | 0.918 m | 170.30° | 314.33% |
| `20260722` xy | actual qpos | 0.0722 rad | **0.0165 m** | 51.06° | 97.42% |
| `20260722` xy | SONIC hybrid | 0.1125 rad | 0.0187 m | **35.55°** | 100.54% |

这些数值说明：

- `20260720 + xy` 中，SONIC-reference hybrid 明显改善了 scanner 的
  最终位置和姿态，值得通过视频确认“拿起、放入抽屉、合上抽屉”的完整
  接触过程；最终下层抽屉 qpos 为 `-0.00022`，原始为 `-0.01278`，所以
  不能仅凭 scanner 终点宣布整个任务成功。
- `20260612` 中，即使使用 xyz assist，pedal 和 lid 仍完全没有复现原始
  运动（任务 qpos 运动量约为零，原始为 1.2438）。因此替换 reference
  姿态没有解决踩垃圾桶的接触问题。
- `20260722 + xy` 的物体终点位置与 qpos-track 接近，姿态有所改善；
  无 assist 时 hybrid 反而产生很大的非预期物体运动。
- hybrid 的身体关节 RMSE 更大。但两列的目标姿态本身不同，所以该
  RMSE 只能描述各自的跟踪难度，不能单独用来判断哪个 controller 更好。

总体上，这次实验支持“SONIC 的上游 reference intent 可以适配到
Teleopit”，但不支持“换 controller 后所有接触任务自然保持成功”。root
assist 仍然是这些实验成功接近原物体轨迹的必要条件，而且 assist 本身是
使用原始 root 轨迹的 oracle，不代表 Teleopit 原生解决了 root 漂移。

## 如何查看

默认 ghost 是原 recording 同一时刻的实际 qpos：

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  Teleopit_rollout/data/20260720_144342_g1_sim_teleopit_sonic_reference_hybrid_root_assist_xy
```

要看这次真正构造给 Teleopit 的 50 Hz hybrid reference：

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  Teleopit_rollout/data/20260720_144342_g1_sim_teleopit_sonic_reference_hybrid_root_assist_xy \
  --ghost-mode reference
```

建议至少回放 `20260720 + xy`、`20260612 + xyz` 和
`20260722 + xy`。前者检查完整的抓取/放入/关抽屉过程；中间一组确认脚与
垃圾桶的接触为何没有产生 pedal/lid 运动；最后一组确认物体运动是否属于
正式任务，而不只是偶然碰撞。

## 验证

```bash
Teleopit_rollout/.venv/bin/python -m unittest discover \
  -s Teleopit_rollout -p 'test_*.py' -v
```

当前共 8 个测试通过，覆盖原 qpos 路径回归、CLI 默认值、自动命名、
hybrid 分量映射、`policy_seq` 对齐、首个 167 维观测/动作/历史形状以及
NPZ provenance。
