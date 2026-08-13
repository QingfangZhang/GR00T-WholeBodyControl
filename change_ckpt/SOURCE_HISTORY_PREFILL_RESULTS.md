# 20260612_144154 source-history prefill 实验

> 兼容说明：本报告记录的是 launcher 统一 offset/默认 prefill 之前生成的历史
> 实验。文中旧命令把 raw 第一组记为 offset 0，旧目录也包含
> `start_offset_*_source_history_prefill`。当前 launcher 已改为“raw 第二组 =
> 公开 offset 0”，默认启用 prefill，且新目录名不再包含 offset/prefill；复现时
> 以当前 README 和新 `launch_manifest.json` 为准。

## 实验问题

本实验检查：regular decoder 在新进程启动时缺少 10 帧状态历史、因而用 0
填充，是否是 `20260612_144154` 启动跟踪误差、root drift 和任务交互失败的
重要原因。

`--start-policy-offset 10` 使用 zero-based unique-`policy_seq` 语义，即从第
11 个 unique group、`policy_seq=21542` 开始。记录的第一个 group 是采集开始前
已经运行到一半的残缺 group，无法恢复它对应的 measured-state 采样时刻；因此
若从 offset 9 开始，最老的一帧历史并不在 CSV 内。offset 10 可以可靠恢复此前
九帧 `seq=21533..21541`。

## 实现语义

decoder 的 994 维输入是当前 64 维 token 加十帧历史；每帧历史包含：

- pelvis angular velocity（3 维）；
- body joint q（29 维）；
- body joint dq（29 维）；
- previous raw action（29 维）；
- gravity direction（3 维）。

程序读取每个 policy group 保存的 `policy_received_dof_pos[0:43]`，并在该 group
前 20 ms 的 400 Hz `qpos/qvel` 时间线上寻找原策略真正采到该状态的时刻。十个
状态匹配到 source rows `3,11,19,27,35,43,50,59,67,75`，而 metadata 首次写入
CSV 的 group rows 是 `7,15,23,31,39,47,55,63,71,79`；两者相差约 10.5 ms。
十组拟合的最大关节误差为 `3.18e-6 rad`。

九个旧状态按 oldest-to-newest 插入 StateLogger。第十帧仍是第一次 live DDS
CONTROL 状态，但 MuJoCo 已初始化到相同的 phase-matched source state，并由 C++
逐项核对 q、dq、quaternion 和 angular velocity；该帧的 previous action 使用
CSV 中当前 group 的 `policy_last_action_in`。这样第一次推理看到的是“九个 source
旧状态 + 一个相同的 live current 状态”，而不是十个伪造的重复状态。

官方 `gear_sonic_deploy` 源码和 release binary 均未修改。新增 wrapper 位于
`change_ckpt/source_history_deploy/`，只替换 StateLogger；其余 controller 代码
仍编译自项目中的官方实现。

## 严格 A/B 设置

两组均为 regular checkpoint、200 Hz control、1000 Hz physics、50 Hz policy、
headless、`root_assist=none`：

- `..._source_state_zero_history`：恢复相同 source 初态，但 StateLogger 仍从空
  history 开始；
- `..._source_history_prefill`：再插入上述九个 source history entries。

两组使用同一个 wrapper binary，而不是“官方 binary 对 wrapper binary”。以下
输入经检查完全相同：

- 初始 qpos/qvel JSON，SHA-256
  `459606674a63405937d0fa7df24380f69c1ce1936a487539bb6888d45cf5ff7c`；
- prepared reference，SHA-256
  `0907596d40f9cc74df18139bc7f2e9527e4c94dfe4864053e374b8271cba4525`；
- deploy wrapper，SHA-256
  `36828960e8a483854207669dc95ab538ce52c91c16c8983aa59a916af484ebc0`；
- encoder、decoder、observation config、source CSV；
- 第一次 live q/dq/quaternion/angular velocity 和第一次 live encoder token。

regular encoder 需要 `current..current+45` 的未来帧。正式对照使用
`--chunk-size 64 --lookahead 64`，避免 50 帧 packet 只剩 4 帧异步安装余量时
偶发的 reference hold。最终两组实际消费并保存的 `target_motion.csv` 为相同的
`382 x 36` 数组，且文件 SHA-256 都是
`3f7ea6a84605fb39ea24dc860a9d379b0457628e210ac9841fddaa581431cf8a`。
因此这次正式 A/B 没有 reference 时钟混杂。

日志索引需要特别注意：zero-history 第一次新 action 位于 `action.csv` index 1；
prefill 的 state/action index 0..8 是九个 source entries，index 9 是 live current
state，第一次新 action 位于 index 10。`token_state.csv` 不包含九个 prefill 数据
行，因此两份 token CSV 的第一条数据分别标记为 index 0 和 index 9；这两个
first-live token 逐值完全相同。

## 运行有效性

| 指标 | zero history | source-history prefill |
|---|---:|---:|
| samples / simulated time | `1426 / 7.125 s` | `1426 / 7.125 s` |
| RTF | `0.999970` | `0.999967` |
| 最大 schedule lag | `3.822 ms` | `5.806 ms` |
| deadline rebase | `0` | `0` |
| publisher late tick | `0` | `0` |
| fallen / invalid state | 否 / 否 | 否 / 否 |
| wall-clock timing valid | 是 | 是 |

## 结果

### 1. decoder 冷启动误差大幅下降

第一次新 decoder action 与源 CSV `policy_seq=21542` 的
`policy_raw_action_out[0:29]` 比较：

| 指标 | zero history | source-history prefill |
|---|---:|---:|
| RMSE | `0.817313` | `0.051266` |
| MAE | `0.651604` | `0.035382` |
| 最大绝对误差 | `1.835536` | `0.132531` |

首 action RMSE 降低 `93.7%`，约为原来的 `1/15.9`。两组 current state 和
current token 完全相同，所以这个差异来自 decoder history，而不是初始 qpos、
encoder 或 reference。

prefill 仍不能逐值复现源 action。共同的 live token 相对源 CSV token 的 RMSE
为 `0.0234375`、最大误差 `0.0625`，64 维中有 9 个量化 bin 不同；此外原采集
时的仿真求解器和控制相位也没有被恢复。

### 2. 启动 tracking 和 root drift 显著改善，但没有完全消失

下表将新 rollout 与原记录同一物理时刻的实际 qpos 比较。body/leg/arm 是逐关节
RMSE，root-XY 是二维距离的 RMS；active 区间为 source 仍推进的
`0..6.125 s`，不包含随后 1 s 的末帧 hold。

| 指标 | zero history | source-history prefill | 改善 |
|---|---:|---:|---:|
| 前 0.2 s body RMSE | `0.13311 rad` | `0.00531 rad` | `96.0%` |
| 前 0.2 s leg RMSE | `0.18360 rad` | `0.00534 rad` | `97.1%` |
| 前 0.2 s root-XY RMS | `1.300 cm` | `0.085 cm` | `93.5%` |
| 前 1.0 s body RMSE | `0.07146 rad` | `0.00867 rad` | `87.9%` |
| 前 1.0 s leg RMSE | `0.09685 rad` | `0.00996 rad` | `89.7%` |
| 前 1.0 s root-XY RMS | `2.099 cm` | `0.153 cm` | `92.7%` |
| active body RMSE | `0.04264 rad` | `0.02791 rad` | `34.5%` |
| active leg RMSE | `0.05959 rad` | `0.03951 rad` | `33.7%` |
| active root-XY RMS | `11.97 cm` | `4.48 cm` | `62.6%` |
| active 末端 root-XY 误差 | `13.33 cm` | `7.88 cm` | `40.9%` |

这说明 zero padding 不只是前 0.2 s 的视觉瞬态：初始错误经过闭环动力学累积，
会明显改变后续 root 和接触轨迹。不过 prefill 后仍有约 8 cm 的末端 root 误差，
所以 decoder 冷启动不是 root drift 的唯一来源。

### 3. 垃圾桶从“完全没有交互”变为“踩动并打开”，但没有保持到末尾

| 物体状态 | zero history | source-history prefill |
|---|---:|---:|
| pedal 最小角度 | 约初始值，未越过 `-0.02` | `-0.219624` |
| lid 最大角度 | 约初始值，未越过 `0.1` | `1.274295` |
| `pedal < -0.1` | 从未发生 | `4.500..5.935 s` |
| `lid > 0.5` | 从未发生 | `4.490..5.955 s` |

zero-history 轨迹中的两个垃圾桶关节在数值精度内保持不动，即没有可观测的机械
交互。prefill 轨迹明确驱动踏板和桶盖发生了符合任务的运动，机器人也始终未
跌倒。因此在这份记录上，恢复 history 不仅改善数值 tracking，也改变了任务
交互结果。CSV 没有保存接触 pair 或接触力，所以“脚与踏板直接接触”是由物体
运动推断的，并非接触力日志验证。

但这还不是原轨迹的完整复现。在 active 末端，source 仍约为
`[pedal=-0.1178, lid=0.6832]`，prefill 已回到约
`[pedal=0.0031, lid=-0.0182]`；也就是比原轨迹提前松开，没有维持末端接触。

## 结论

`20260612_144154` 的结果支持这个判断：新进程中 decoder history 全零是 regular
rollout 与原轨迹差异的一个重要原因。source-history prefill 将首 action 和前
0.2 s tracking 误差降低约一个数量级以上，并使实验从完全没有垃圾桶交互变为
成功踩动踏板、打开桶盖。

它不是完整复现方案。剩余差异还包括 current encoder token 的量化差异、未恢复
的旧 ctrl/contact/`qacc_warmstart`/solver state、原采集 DDS 相位，以及 SONIC
对 global translation 的有限约束；这些会继续造成 root drift 和提前失去接触。

## 输出与 replay

```text
change_ckpt/data/20260612_144154_g1_sim_regular_start_offset_10_source_state_zero_history
change_ckpt/data/20260612_144154_g1_sim_regular_start_offset_10_source_history_prefill
```

对照查看：

```bash
.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt/data/20260612_144154_g1_sim_regular_start_offset_10_source_state_zero_history \
  --ghost-mode source

.venv_replay/bin/python change_ckpt_track/replay_mujoco_compare.py \
  change_ckpt/data/20260612_144154_g1_sim_regular_start_offset_10_source_history_prefill \
  --ghost-mode source
```
