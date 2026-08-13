# Controller Replacement（统一确定性实验）

这个目录用于回答一个明确的问题：在保持同一任务场景、同一参考动作和同一手指目标的条件下，把数据采集时的全身 controller 换成另一个 controller，机器人是否仍能稳定接触物体并完成任务，生成的新轨迹是否仍可作为训练数据候选。

原有的 `change_ckpt/`、`change_ckpt_track/`、`Teleopit_rollout/` 和官方源码没有被这个流程替代；这里是一套独立、单进程、确定性的正式比较实现。

## 固定实验协议

| 项目 | 固定设置 |
|---|---:|
| MuJoCo physics | 2000 Hz（每步 0.5 ms） |
| controller 原生 PD | 200 Hz |
| policy inference | 50 Hz |
| `data.csv` 状态记录 | 400 Hz |
| 每次 policy 对应 | 40 physics steps / 4 PD updates / 8 CSV rows |
| decoder/history 初始化 | source-history prefill（默认且正式实验唯一支持） |
| self-warmup | 新 controller 前 10 次推理 |
| tracking 指标 | 从第 11 次新 controller 推理开始 |
| root assist | `none` 或 `xy`；不提供 `xyz` |

400 Hz 在这里仅是保存频率，不是 policy 或 PD 频率。每行数据固定推进 5 个 MuJoCo physics step；reference、policy 和 physics 都由整数 step 决定，不依赖墙上时间、DDS 最新值或 ZMQ 消息。因此机器变慢只会延长运行耗时，不会让 reference/ghost 与机器人走不同的时间轴。

Viewer 也不改变实验轨迹：它只显示并按真实时间 sleep，不能跳帧、追赶或选择新的 reference。

## Controller 与 reference

支持四种 controller：

- `regular`：SONIC release，原生 step-5 reference、994 维 decoder history 和 SONIC PD。
- `low_latency`：SONIC low-latency，原生 consecutive step-1 reference。
- `sonic_v1_1`：SONIC v1.1，原生 heading-normalized orientation。
- `teleopit`：Teleopit v0.5 `track_g1.onnx`，原生 167 维 observation、10 帧 observation history、action mapping 和 PD。

支持两种明确分开的 reference：

- `reference_motion`（正式主实验）：保留 recording 中 SONIC 的动作意图。对 Teleopit 使用显式 hybrid：reference body joints + reference pelvis orientation + recording 的实际 root xyz（因为 SONIC `reference_motion` 没有 root xyz）。
- `executed_qpos`（辅助 ablation）：从 recording 的实际机器人 qpos 构造目标；这就是原 `change_ckpt_track` 思路。

SONIC regular/v1.1 使用记录的十个 regular slots；low-latency 使用连续十个 50 Hz slots。不会把两者偷偷当成相同的 future window。

## Source-history 和 warmup 的准确含义

默认 `--policy-offset 10` 是 edge-trim 后 reference 序列中的 offset。程序先取该帧的实际 `policy_seq`，再反查 raw CSV policy group；因此第一组是否被截断都不会造成 offset 歧义。

接管时：

1. 从原 recording 严格恢复前 9 个 policy state/action；
2. 用 `policy_received_dof_pos` 做相位匹配，得到可能落在两个 400 Hz 行之间的 takeover 时间；
3. MuJoCo 用该时刻的精确 `qpos/qvel` 初始化；
4. 第一个 live state 成为 history 的第 10 帧；
5. 新 controller 从此开始连续控制。第 1～10 次新 controller action 全部真实作用于机器人和物体，也全部保存；仅 tracking 汇总在第 11 次推理开始。

Teleopit 的 previous action 不能直接复制 SONIC raw action。程序先把旧 SONIC action 转成物理 `q_target`，再反解成 Teleopit raw-action 坐标；转换、clip 状态和 residual 都保存在 metadata 中。

不提供“补零”“重复当前帧”或“只填 state、不填 action”的正式运行选项。

## 运行环境

当前可直接使用的统一环境是：

```bash
Teleopit_rollout/.venv/bin/python
```

它同时包含 MuJoCo 3.10 和 ONNX Runtime。当前已安装的是 CPU provider，所以下面的例子使用 CPU。SONIC 如果要使用 `--device cuda`，该环境必须另行安装并确认 ONNX Runtime CUDA provider；Teleopit 正式命令当前使用 CPU。

## 单个实验命令

主实验（SONIC regular、无 root assist）：

```bash
Teleopit_rollout/.venv/bin/python controller_replacement/launch_rollout.py \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  --controller regular \
  --reference-mode reference_motion \
  --root-assist none
```

Low-latency + root assist xy：

```bash
Teleopit_rollout/.venv/bin/python controller_replacement/launch_rollout.py \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  --controller low_latency \
  --reference-mode reference_motion \
  --root-assist xy
```

SONIC v1.1：

```bash
Teleopit_rollout/.venv/bin/python controller_replacement/launch_rollout.py \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  --controller sonic_v1_1 \
  --reference-mode reference_motion \
  --root-assist none
```

Teleopit：

```bash
Teleopit_rollout/.venv/bin/python controller_replacement/launch_rollout.py \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  --controller teleopit \
  --reference-mode reference_motion \
  --root-assist xy
```

`executed_qpos` ablation 只需改：

```bash
--reference-mode executed_qpos
```

短冒烟测试可以加 `--policy-count 12`；正式任务不要限制 `policy-count`。实时显示可加 `--viewer`，但正式批量建议 no-viewer。

默认输出目录为：

```text
controller_replacement/data/<recording>_<controller>_<reference-mode>[_root_assist_xy]
```

重复同一条件会覆盖同名结果。为防止误删，程序只会自动删除带有自身 marker 的目录；如果 `--output` 指向普通用户目录，会拒绝执行。

## 批量运行

对一个或多个 recording 依次运行四个 controller：

```bash
controller_replacement/run_experiments.sh \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  sample_data/ztj/20260612/20260720_144342_g1_sim
```

切换条件：

```bash
ROOT_ASSIST=xy REFERENCE_MODE=reference_motion \
  controller_replacement/run_experiments.sh <recording> [recording ...]
```

某个实验失败时脚本会记录 `FAILED` 并继续剩余实验，最后返回非零状态；不会像 `set -e` 脚本那样在第一个失败处静默停止。

## 输出内容

每个成功目录包含：

| 文件 | 内容 |
|---|---|
| `data.csv` | 与该 source recording 完全相同的 header/列顺序；400 Hz 新物理轨迹 |
| `policy_telemetry.npz` | 50 Hz controller-native observation/history/action/target/torque，以及 controller 实际消费的 reference |
| `source_timeline.npz` | 每个 400 Hz 输出样本对应的精确 source 时间、插值 qpos 和最近 provenance 行 |
| `contact_telemetry.npz` | 400 Hz robot/world/environment contact 数量与法向力摘要 |
| `prepared_reference.npz` | 50 Hz reference 与 source policy 边界映射 |
| `metrics.json` | warmup 后 tracking、root、力矩饱和、稳定性和任务物体 qpos 变化 |
| `run_manifest.json` | controller/model SHA-256、reference、频率、root assist、协议元数据 |
| `data_schema.json` | 完整 CSV header 及字段组语义 |
| `source_history_prefill.json` | 实际使用的原生 source-history payload |
| `source_history_context.json` | raw/processed offset、phase match 和 Teleopit action 转换 provenance |
| `model_snapshot/` | 当前任务场景 XML 及资产链接，可直接 replay |
| `run_complete.json` | 仅完整成功后生成 |

### `data.csv` 中 controller 字段

- SONIC：保存真实 64 维 token（其余 token capacity 清零）、真实 last/raw action、43 维 received state；这些值在同一次 50 Hz inference 的 8 个 400 Hz 行上保持。
- SONIC `reference_motion[0:640]`：保存该次 encoder 实际使用的 q/dq/相对 orientation（包含实时 robot orientation 的影响），`[640:1024]` 清零。
- `policy_telemetry.npz` 的 `reference` 也使用同一份 controller 实际输入：SONIC 为实际 640 维 encoder reference，Teleopit 为实际 36 维 reference pose。
- 非 SONIC：所有 SONIC-only token/last/raw/received/reference 字段和 size/valid 均显式填 0，不复制旧 checkpoint 的假数据。
- Teleopit 的 167 维 observation、10×167 history 和 native action 全部在 `policy_telemetry.npz`，不会硬塞进 SONIC 列名。
- `qpos/qvel`、手目标字段和任务物体状态都来自新 rollout；实际手 PD torque 记录在 native telemetry，legacy `hand_tau` 仍保持 feed-forward torque 的原语义（这里为 0）。

因此 `data.csv` 可以继续被现有 replay/数据工具读取，但做训练或分析时应同时读取 `data_schema.json` 和 `policy_telemetry.npz`，尤其是非 SONIC 结果。

## 查看结果

只看实心机器人：

```bash
./sample_data/ztj/replay_mujoco_csv \
  controller_replacement/data/<result-folder>
```

同时显示原 recording 的实际 qpos ghost（默认 cyan body + amber hands）：

```bash
.venv_replay/bin/python controller_replacement/replay_mujoco_compare.py \
  controller_replacement/data/<result-folder>
```

先做无 GUI 校验：

```bash
.venv_replay/bin/python controller_replacement/replay_mujoco_compare.py \
  controller_replacement/data/<result-folder> --dry-run
```

这个 wrapper 使用 `source_timeline.npz` 中逐样本、phase-matched 的插值 source qpos；不会假设原 CSV 的每个 `policy_seq` 恰好都有 8 行。

## 如何判断“数据是否可用”

自动指标只负责可重复的事实：轨迹是否完整、是否出现 NaN、body/root tracking、torque saturation、root 高度、robot-environment contact 以及任务物体各 qpos 的起点/终点/最大变化。`robot-environment` 会包含物体、家具或固定装置，不能仅凭“发生接触/物体动了”自动宣称任务成功。

正式判断建议同时满足：

1. `run_complete.json` 存在；
2. `metrics.json` 没有数值异常或明显失稳；
3. 用 compare viewer 检查 body/root/手的偏差；
4. 根据任务定义人工或用后续 task-specific evaluator 标注成功（例如物体确实进入抽屉且抽屉合上，而不是只发生碰撞）；
5. 分开报告 `root-assist none` 和 `root-assist xy`，不能把 assisted 结果当作 controller 独立 root tracking 能力。

## 测试

```bash
Teleopit_rollout/.venv/bin/python -m unittest discover \
  -s controller_replacement/tests -v
```

测试覆盖 reference 构造、raw/processed offset、phase-matched 400 Hz 插值、SONIC 三个 ONNX、recorded token exact-match、Teleopit 官方 167 维 observation 对照、source-history、CSV/telemetry 字段、MuJoCo 原生 PD 和输出保护。
