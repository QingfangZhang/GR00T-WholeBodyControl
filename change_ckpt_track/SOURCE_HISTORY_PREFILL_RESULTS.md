# qpos-track source-history prefill 实验结果

> 兼容说明：本报告记录的是 launcher 统一 offset/默认 prefill 之前生成的历史
> 实验。文中旧 offset 是 qpos processed index，旧目录也包含
> `start_offset_*_source_history_prefill`。当前 launcher 已改为“raw 第二组 =
> 公开 offset 0”，默认启用 prefill，且新目录名不再包含 offset/prefill；复现时
> 以当前 README 和新 `launch_manifest.json` 为准。

日期：2026-08-12

## 实验条件

- checkpoint：regular
- reference：原 recording 的机器人 qpos 构造的 50 Hz reference
- encoder future window：canonical `[0,5,10,...,45]`
- decoder 启动历史：原 CSV 前九个 50 Hz 实测状态 + 第一条 live MuJoCo 状态
- root assist：none
- simulator：200 Hz、无 viewer
- publisher：50 Hz，chunk-size/lookahead 均为 100

这里的 processed offset 是经过 `policy_valid` 过滤和首尾截边后的 qpos-reference
下标；raw offset 是原 CSV 连续 `policy_seq` group 的下标。为了与此前
`change_ckpt` 的 raw offset 11 保持同一个真实起点，不同 recording 的 processed
offset 可能是 10 或 11。

## 启动映射与运行有效性

| recording | processed | policy_seq | raw | samples | RTF | max lag (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 20260720_144342 | 10 | 31126 | 11 | 2457 | 0.999952 | 21.870 |
| 20260612_144127 | 10 | 20173 | 11 | 1255 | 0.999965 | 24.964 |
| 20260612_144154 | 11 | 21543 | 11 | 1422 | 0.999907 | 4.142 |
| 20260612_144214 | 11 | 22503 | 11 | 1178 | 0.999875 | 17.593 |
| 20260722_154958 | 11 | 73327 | 11 | 1988 | 0.999979 | 8.980 |
| 20260722_145020 | 11 | 17209 | 11 | 1780 | 0.999976 | 8.337 |
| 20260722_160121 | 10 | 107495 | 11 | 1704 | 0.999977 | 9.066 |

七份正式结果均满足：source CSV 正常结束、`wall_clock_timing_valid=true`、
deadline rebase 为 0、无跌倒、无非有限状态、hand overrun 为 0、publisher
`late_ticks=0`。deploy 均确认载入九条 source history；第一条 live state 的
quaternion / angular velocity / q / dq 最大误差打印值均为 0。

`20260612_144214` 第一次运行发生一次 hand publisher overrun，已作为无效运行
丢弃并同名覆盖；表中是第二次有效结果。`20260612_144154` 也覆盖了此前一次
`wall_clock_timing_valid=false` 的非正式运行。

## Tracking 与任务物体物理指标

| recording | body RMSE (rad) | root XY RMS (m) | yaw RMS (deg) | 任务物体指标 |
|---|---:|---:|---:|---|
| 20260720_144342 | 0.06025 | 0.15457 | 1.473 | 抽屉完成约 1.039 倍原轨迹进度；scanner 位移仅 0.000022 m |
| 20260612_144127 | 0.04922 | 0.04329 | 1.022 | 垃圾桶踏板和盖子基本无运动 |
| 20260612_144154 | 0.05664 | 0.05328 | 1.744 | 垃圾桶踏板和盖子基本无运动 |
| 20260612_144214 | 0.05732 | 0.04902 | 3.261 | 垃圾桶踏板和盖子基本无运动 |
| 20260722_145020 | 0.04743 | 0.04743 | 0.974 | can 位移 0.276 m；终点位置距原轨迹 0.048 m |
| 20260722_154958 | 0.04899 | 0.02102 | 1.105 | can 位移 0.718 m，最终 z=0.158 m；终点位置距原轨迹 0.839 m |
| 20260722_160121 | 0.03562 | 0.02308 | 0.771 | can 位移 0.192 m；终点位置距原轨迹 0.024 m |

任务物体 qpos 是物理终态指标，不自动等价于语义成功。根据物理量，
`20260720_144342` 没有拿起 scanner，三个垃圾桶实验也没有压动踏板；
`20260722_160121` 的物体终点最接近原轨迹。抓取、稳定接触和完整任务是否成功，
仍应使用 `replay_mujoco_compare.py` 逐条目视确认。

## 与旧零历史结果的解释边界

现有旧 regular 结果从 processed offset 0 启动，而本实验从对应 raw offset 11
启动；两者少了开头约 0.22 秒，并使用不同的初始物理相位，所以只能做趋势比较，
不能把差异全部归因于 history prefill。

在具有旧基线的五份 recording 中，prefill 后 body RMSE 在四份下降约
4.3%--11.4%，在 `20260720_144342` 上上升约 1.2%；但任务接触没有因此稳定
改善，`20260612_144214` 甚至从旧结果的明显垃圾桶运动变成几乎无运动。这说明
补齐 decoder 启动历史能改变并经常略微改善关节 tracking，但不能单独解决 root
漂移或保证接触任务成功。要严格分离“起始相位”和“历史预填”的作用，还需要在
完全相同的 processed offset 和 phase-matched 初态下再跑一组 zero-history 对照。
