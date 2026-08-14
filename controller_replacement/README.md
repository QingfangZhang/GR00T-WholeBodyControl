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
| 手部力矩上限 | 默认 `sonic_release=[2.45, 0.7×6] Nm` |
| fall 判定 | `qpos[2] < 0.2 m` 后锁存，但不停止仿真 |
| inference backend | Python 内嵌 ONNX Runtime（不使用 DDS/ZMQ） |

SONIC 保留 release 的 action scale 和 Kp/Kd，但 release C++ 并不发布额外的 controller-side body torque cap；其实际 body PD 力矩由每个 recording 暂存的 MuJoCo 场景 actuator `ctrlrange` 限制（这批 G1 场景的踝关节为 50 Nm）。Teleopit 则继续使用其官方 tracker 的原生 body torque limits，并与场景 actuator 上限取更严格者。手部两套 controller 共用正式的 SONIC release 限幅；`--hand-torque-profile staged_xml` 只用于保留 recording XML 约 1.4 Nm 上限的 sensitivity 实验。

两个 controller 的角速度信号也明确分开：SONIC 使用 release MuJoCo bridge 的 free-joint `qvel[3:6]`；Teleopit 使用 `mj_objectVelocity(pelvis, local=1)` 得到的 pelvis-link 局部角速度。程序不会再把二者混成一个含义含糊的 `root_ang_vel`。

400 Hz 在这里仅是保存频率，不是 policy 或 PD 频率。每行数据固定推进 5 个 MuJoCo physics step；reference、policy 和 physics 都由整数 step 决定，不依赖墙上时间、DDS 最新值或 ZMQ 消息。因此机器变慢只会延长运行耗时，不会让 reference/ghost 与机器人走不同的时间轴。

`root-assist xy` 会在半开执行区间 `[0,T)` 内、每个有后续 physics interval 的 200 Hz PD 边界，把机器人 free root 的 `qpos[:2]` 覆盖为 phase-matched source 的 x/y，并同时把 `qvel[:2]` 覆盖为 source 的 x/y 线速度；隐藏终点 `T` 只用于统计，不再额外 teleport。source 速度耗尽时速度显式置零。`metrics.json` 会分别记录 source-velocity 和 zero-fallback 的注入次数。力矩最大值和饱和率也按每个 200 Hz PD update 统计，而不是每个 50 Hz policy 只抽样一次。

Viewer 也不改变实验轨迹：它重建独立的 MuJoCo model/data，只把 live state 单向复制到显示 clone，不能修改正式 physics；显示按真实时间 sleep，不能跳帧、追赶或选择新的 reference。带 `--viewer` 的自动输出目录会增加 `_viewer` 后缀，避免与 no-viewer 结果互相覆盖。

## Controller 与 reference

支持四种 controller：

- `regular`：SONIC release，逐帧使用 recording 已保存的十个 regular slots、994 维 decoder history 和 SONIC PD；不会假设这些 slots 必然是 `0,5,...,45`。
- `low_latency`：SONIC low-latency，原生 consecutive step-1 reference。
- `sonic_v1_1`：SONIC v1.1，原生 heading-normalized orientation。
- `teleopit`：Teleopit v0.5 `track_g1.onnx`，原生 167 维 observation、10 帧 observation history、action mapping 和 PD。

支持两种明确分开的 reference：

- `reference_motion`（正式主实验）：保留 recording 中 SONIC 的动作意图。对 Teleopit 使用显式 hybrid 来做 FK：reference body joints + reference pelvis orientation + recording 的实际 root xyz（因为 SONIC `reference_motion` 没有 root xyz）；tracker 最终实际消费的是 torso-link 相对朝向、速度、重力和高度，并不消费绝对 root xy。
- `executed_qpos`（辅助 ablation）：从 recording 的实际机器人 qpos 构造目标；这就是原 `change_ckpt_track` 思路。

SONIC regular/v1.1 使用记录的十个 regular slots；low-latency 使用连续十个 50 Hz slots。不会把两者偷偷当成相同的 future window。

## Source-history 和 warmup 的准确含义

默认 `--raw-policy-group-offset 11` 使用原始 `data.csv` 中 zero-based 的 raw `policy_seq` group 坐标：固定选择第 12 个 raw group。程序再用该 group 的准确 `policy_seq` 反查 edge-trim 后 reference 序列中的位置；因此第一组是否完整、是否被 reference loader 裁掉，都不会改变接管的 raw group。为避免再混淆 raw 和 processed 坐标，不再使用含义不明确的 `--policy-offset`。

接管时：

1. 从原 recording 严格恢复前 9 个 policy state/action；
2. 用 `policy_received_dof_pos` 做相位匹配，得到可能落在两个 400 Hz 行之间的 takeover 时间；
3. MuJoCo 用该时刻的精确 `qpos/qvel` 初始化；
4. 第一个 live state 成为 history 的第 10 帧；
5. 新 controller 从此开始连续控制。第 1～10 次新 controller action 全部真实作用于机器人和物体，也全部保存；仅 tracking 汇总在第 11 次推理开始。

Teleopit 的 previous action 不能直接复制 SONIC raw action。程序先把旧 SONIC action 转成物理 `q_target`，再反解成 Teleopit raw-action 坐标；转换、clip 状态和 residual 都保存在 metadata 中。source-history 中的 `qvel[3:6]` 也会通过同一份 pinned Teleopit robot model 重建成 pelvis-local 角速度，使前 9 帧和 live 帧使用同一 native 信号定义。

不提供“补零”“重复当前帧”或“只填 state、不填 action”的正式运行选项。

timeline 会依据实际 MuJoCo joint type，对机器人和任务物体的全部 free/ball quaternion 使用 shortest-arc SLERP 并归一化；不会对任务物体姿态做普通线性插值。若 recording 末尾不足完整的 8 个 400 Hz row，最后那个不完整 20 ms policy interval 会被明确丢弃，requested/executed 数量写入 manifest 和 completion certificate。

## 运行环境

当前可直接使用的统一环境是：

```bash
Teleopit_rollout/.venv/bin/python
```

它同时包含 MuJoCo 3.10 和 ONNX Runtime。当前已安装的是 CPU provider，所以下面的例子使用 CPU。SONIC 或 Teleopit 若使用 `--device cuda`，该环境必须另行安装并确认 ONNX Runtime CUDA provider；程序在 CUDA EP 不存在时会明确失败，不会静默退回 CPU。

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

正式默认手部 profile 和 fall threshold 无需写参数。手部 sensitivity 或自定义 fall threshold 可分别写：

```bash
--hand-torque-profile staged_xml
--fall-height-m 0.25
```

默认输出目录为：

```text
controller_replacement/data/<recording>_<controller>_<reference-mode>_protocol2[_root_assist_xy]
```

非默认 `staged_xml`、fall threshold、raw offset、`policy-count` 和 CUDA device 都会追加独立后缀，避免与正式默认条件互相覆盖。自定义 Teleopit checkpoint/XML 或自定义 asset root 必须显式给出 `--output`。

已有的不带 `_protocol2` 后缀的旧结果会原样保留，新协议不会自动覆盖它们。

重复同一条件会覆盖同名结果，但只允许覆盖 marker、`run_manifest.json`、`run_complete.json` 都完整的旧结果。发布使用同名非阻塞文件锁、唯一 work/backup 目录和失败恢复；并发写同一结果会立即拒绝。若自动名字与另一个同 basename recording 冲突，程序也会拒绝并要求显式 `--output`。输出与 recording、模型或仓库根发生危险路径重叠时不会执行。

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
HAND_TORQUE_PROFILE=sonic_release FALL_HEIGHT_M=0.2 \
  controller_replacement/run_experiments.sh <recording> [recording ...]
```

某个实验失败时脚本会记录 `FAILED` 并继续剩余实验，最后返回非零状态；不会像 `set -e` 脚本那样在第一个失败处静默停止。

## 输出内容

每个成功目录包含：

| 文件 | 内容 |
|---|---|
| `data.csv` | 与该 source recording 完全相同的 header/列顺序；400 Hz 新物理轨迹 |
| `policy_telemetry.npz` | 50 Hz controller-native observation/history/action/target/torque，以及 controller 实际消费的 reference |
| `source_timeline.npz` | 每个 400 Hz 输出样本对应的精确 source 时间、插值 qpos、最近 provenance 行和 pre-fall mask |
| `contact_telemetry.npz` | 400 Hz robot/world/environment contact 数量与法向力摘要 |
| `prepared_reference.npz` | 完整的 50 Hz controller-neutral reference 数组、两套 joint order、optional source 数组和 provenance；pickle-free 重载时会校验同内容 source `data.csv` 的 SHA-256 |
| `metrics.json` | warmup 后且 fall 前的 tracking、phase-matched source root、controller 实际输入的 anchor 朝向/可用高度、力矩饱和；任务物体和隐藏终点统计 |
| `run_manifest.json` | controller/model SHA-256、源码/scene/compiled-MuJoCo 指纹、运行环境、reference、频率和协议元数据 |
| `data_schema.json` | 完整 CSV header 及字段组语义 |
| `source_history_prefill.json` | 实际使用的原生 source-history payload |
| `source_history_context.json` | raw policy-group offset、解析后的准确 `policy_seq`、phase match 和 Teleopit action 转换 provenance |
| `model_snapshot/` | 当前任务场景 XML 及资产链接，可直接 replay |
| `run_complete.json` | 最后生成；固定步调度/有限状态/fall/任务成功资格及全部主要输出 SHA-256 |

成功目录先在同级随机 work directory 中完整生成并认证，再在 per-output lock 下发布。自动命名的重复运行只有在 source/model/CLI 条件、Viewer 条件及当前 compiled MuJoCo scene（含实际加载资产）都一致时才能覆盖；不同条件必须显式指定另一个 `--output`。

### `data.csv` 中 controller 字段

- SONIC：保存真实 64 维 token（其余 token capacity 清零）、真实 last/raw action、43 维 received state；这些值在同一次 50 Hz inference 的 8 个 400 Hz 行上保持。
- SONIC `reference_motion[0:640]`：保存该次 encoder 实际使用的 q/dq/相对 orientation（包含实时 robot orientation 的影响），`[640:1024]` 清零。
- `policy_telemetry.npz` 的 `reference` 也使用同一份 controller 实际输入：SONIC 为实际 640 维 encoder reference，Teleopit 为实际 36 维 reference pose。
- 非 SONIC：SONIC-only token/last/raw/reference 字段和 size/valid 均显式填 0，不复制旧 checkpoint 的假数据；但 controller-neutral 的 `policy_received_dof_pos[0:43]` 保存真实 `body29+left7+right7` measured state，并在同一次推理的 8 行中保持。
- Teleopit 的 167 维 observation、10×167 history 和 native action 全部在 `policy_telemetry.npz`，不会硬塞进 SONIC 列名。
- `qpos/qvel`、`left_hand_q/right_hand_q`（rate-limited applied target）和任务物体状态都来自新 rollout；`left_hand_close/right_hand_close` 是无法由 controller 重建的 source passthrough，`data_schema.json` 会明确标记。50 Hz telemetry 分开保存 raw desired hand target 与真正用于该 PD update 的 applied target；实际手 PD torque 也在 telemetry，legacy `hand_tau` 仍保持 feed-forward torque 的原语义（这里为 0）。

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

这个 wrapper 使用 `source_timeline.npz` 中逐样本、phase-matched 的精确插值 source qpos；不会退化成 nearest-row ghost，也不会假设原 CSV 的每个 `policy_seq` 恰好都有 8 行。打开 Viewer 前会校验 `run_complete.json` 对 `data.csv`、sidecar 和 manifest 的 SHA-256，校验原 source CSV SHA，并默认使用 manifest 记录的 asset root 和 snapshot XML。MuJoCo 版本相同时还会比较完整 MJB fingerprint；若 rollout/replay 分别使用 3.10/3.2，会明确提示 MJB 字节因版本不同不能直接比较。`model_snapshot/` 的大资产仍是外部链接，因此移动结果时也要保留记录的资产目录，或显式传 `--asset-model-root`。

## 如何判断“数据是否可用”

自动指标只负责可重复的事实：轨迹是否完整、是否出现 NaN、body/root tracking、torque saturation、root 高度、robot-environment contact 以及任务物体各 qpos 的起点/终点/最大变化。root tracking 会分别报告“机器人 vs 同一仿真时刻的 source executed root”和“controller 实际消费的相对 anchor rot6d 所代表的朝向误差”，不会把二者混成一个 target。Teleopit 另报告 torso anchor 高度误差；SONIC 和 Teleopit 都不把不存在于网络输入中的绝对 reference root xy 伪装成 tracking target。SONIC v1.1 的误差直接来自 heading-normalized encoder 输入，因此不会额外计入已被 normalization 去掉的 robot roll/pitch。

一旦 `qpos[2]` 在任一 2000 Hz physics substep 低于 threshold，fall 会永久锁存，root-z min/max 也使用同一 2000 Hz 采样；之后的样本仍保存，物体/接触也继续真实演化到固定任务终点，但 tracking 汇总只使用 fall 前样本，且该轨迹的 `task_success_eligible=false`。`robot-environment` 会包含物体、家具或固定装置，不能仅凭“发生接触/物体动了”自动宣称任务成功。

正式判断建议同时满足：

1. `run_complete.json` 存在；
2. `metrics.json` 没有数值异常或明显失稳；
3. 用 compare viewer 检查 body/root/手的偏差；
4. 根据任务定义人工或用后续 task-specific evaluator 标注成功（例如物体确实进入抽屉且抽屉合上，而不是只发生碰撞）；
5. 分开报告 `root-assist none` 和 `root-assist xy`，不能把 assisted 结果当作 controller 独立 root tracking 能力。

## ORT backend 固定输入复核

SONIC rollout 会在 telemetry 保存每次推理的精确 encoder/decoder 输入。可固定抽取第 `0、10、middle、last` 次推理并重新运行 ORT：

```bash
Teleopit_rollout/.venv/bin/python controller_replacement/backend_parity.py \
  controller_replacement/data/<sonic-result-folder> \
  --backend ort --device cpu \
  --output-directory controller_replacement/analysis/<result-folder>_ort_cpu
```

输出的 `backend_parity_cases.npz` 和 `backend_parity.json` 会分别比较 token、固定 decoder-input action、端到端 action 与重复运行确定性，并由 `backend_parity_complete.json` 绑定两者 SHA。三者先在临时目录完成，再作为一个受锁目录发布，不会留下新旧文件混合 pair；同名结果还会绑定 ORT 版本、实际 encoder/decoder provider 和实现类型，运行时条件改变后必须换输出目录。这些是派生分析文件，不写入已经由 `run_complete.json` 认证的 rollout 目录；省略 `--output-directory` 时也会自动写到 rollout 的同级 `analysis/`。只有完整、有限、fixed-step 的 protocol-2 rollout 才能生成正式 parity。这里不把 ORT 结果冒充 TensorRT 结果；当前 `--backend tensorrt` 会明确返回 unavailable。以后接入 Python TensorRT 时可以直接复用同一批固定输入，不需要引入 DDS/ZMQ。manifest 同时记录 requested device、session 实际 provider 和环境可用 provider；SONIC 与 Teleopit 的正式 ORT session 都固定为 sequential、intra/inter-op 单线程；请求 `cuda` 时必须实际存在 `CUDAExecutionProvider`，不会静默回退到 CPU。

## 测试

```bash
Teleopit_rollout/.venv/bin/python -m unittest discover \
  -s controller_replacement/tests -v
```

测试覆盖 reference 构造与完整 round-trip、raw policy-group 到 edge-trim reference 的准确映射、phase-matched 400 Hz 插值、SONIC 三个 ONNX、recorded token exact-match、Teleopit 官方 167 维 observation 对照、source-history、两套角速度、43 维 received state、手部 profile、fall/metric 语义、CSV/telemetry 字段、MuJoCo 原生 PD、backend 固定输入复核和输出保护。
