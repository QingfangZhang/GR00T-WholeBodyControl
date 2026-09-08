# SONIC checkpoint 与全身 Controller Replacement 实验：对话历史及技术演进记录

> 整理日期：2026-09-02
> 覆盖范围：本次 Codex 会话中可追溯的讨论、源码检查、数据验证、实验实现、结果分析与 Git 演进。
> 文档性质：技术纪要，不是逐字聊天记录；已省略重复提问、工具调用细节、内部运行日志以及可能包含环境敏感信息的原始 JSONL 内容。

## 0. 阅读说明

这份文档服务于两个目的：

1. 记录我们是怎样从“能不能用另一个 checkpoint 重放 CSV”逐步发展到“在同一任务场景中闭环替换整个全身 controller”的；
2. 明确区分已经验证的事实、用户通过 replay 得到的观察、当时提出但尚未证实的解释，以及仍然待解决的问题。

全文采用以下标记：

- **[已验证]**：通过源码、模型签名、CSV 数值、SHA-256、自动测试或实际 rollout 检查过；
- **[用户观察]**：用户在 MuJoCo viewer 或录像中观察到，但未必有自动任务成功判定；
- **[用户决策]**：讨论后确定的工程约束或正式实验协议；
- **[推测]**：有依据的解释或研究假设，尚不能当作确定事实；
- **[待解决]**：仍缺源码、数据、重复实验或 task-specific evaluator。

当前各实现自己的操作说明仍应以这些文档为准：

- [`change_ckpt/README.md`](change_ckpt/README.md)
- [`change_ckpt_track/README.md`](change_ckpt_track/README.md)
- [`Teleopit_rollout/README.md`](Teleopit_rollout/README.md)
- [`controller_replacement/README.md`](controller_replacement/README.md)
- [`sim_reference_overlay/README.md`](sim_reference_overlay/README.md)

---

## 1. 一页摘要

最初的问题是：已有的 loco-manipulation 数据由 SONIC regular checkpoint 控制机器人采集，能否把 checkpoint 换成 SONIC low-latency、SONIC v1.1，甚至 Teleopit，并判断机器人是否仍能与物体正确接触、完成原任务，以及新生成的数据是否仍可用于训练 GR00T/VLA。

对话最终形成的核心认识如下。

1. **不能把新 decoder 的 action 直接写成 qpos。**
   raw action 必须经过 controller 自己的 action mapping、PD、关节/力矩限制和 MuJoCo 接触动力学，才能得到新的实际 qpos 和任务物体轨迹。

2. **必须区分两类 reference。**

   - `reference_motion`：原采集系统真正发给 SONIC 的动作意图，是正式主实验；
   - `executed_qpos`：旧 controller 实际执行出来的 qpos，用它重新构造 tracker 目标，是辅助的 qpos-track ablation。

3. **重放原 CSV 与替换 controller 是两回事。**
   原 C++ replay 只是逐行把 CSV qpos 写入 MuJoCo 并显示，不重新运行 policy，也不重新计算接触动力学。

4. **SONIC 的 640 维有效 reference 已被严格验证。**
   它是 `10×29 q + 10×29 dq + 10×6 relative anchor orientation`。CSV 为 reference 预留 1024 列，但有效内容是前 640 维。对一份 regular 数据，重建 encoder 后得到的 64 维 token 与 CSV 中保存的 token 完全一致。

5. **原数据中实际 future slot 不一定是理想的 `[0,5,10,...,45]`。**
   多份 recording 从开始阶段就出现 `[0,5,9,9,...]`、`[0,4,4,...]` 或类似的重复尾槽。它是通过数值匹配推断出的实际 lag，不是 CSV 自带字段；现象已验证，采集端根因尚未确认。

6. **SONIC regular 不闭环跟踪绝对 root xy。**
   初始 root 完全对齐也不保证后续不漂。脚部 tracking、接触、速度误差、历史初始化、PD 和 solver 状态都会让 root 漂移逐步累积。

7. **source-history prefill 明显比零历史合理。**
   新 deploy 进程即使从 recording 第十帧开始，内部 history 仍然是空的。恢复 takeover 前 9 个 source policy state/action，再把第一个 live state 作为第 10 帧，可大幅降低最初 action 和短时 tracking 误差。

8. **root assist 能显著改善接触任务，但它是 oracle。**
   `xy` 在运行时逐个 PD 边界把机器人 root x/y 与 vx/vy 对齐到原 recording。它会改变真实闭环和后续接触，不能当作 controller 自己具备 root tracking 的证据。

9. **Teleopit 可以作为非 SONIC controller 基线。**
   但它不能直接消费 SONIC 640 维 reference，需要把 SONIC pose 适配为 Teleopit 的 torso-link reference 和 167 维 observation；其十帧输入是过去 observation history，不是 SONIC 的十个未来 slot。

10. **正式统一方案是 `controller_replacement/`。**
    它使用单进程固定仿真步：2 kHz physics、200 Hz PD、50 Hz policy、400 Hz CSV，不依赖 DDS/ZMQ 或墙上时间；保留各 controller 的原生 observation、action mapping、PD 和 torque limits；当前推理 backend 是 ONNX Runtime CPU，而不是 TensorRT。

11. **“物体动了”不等于任务成功。**
    新轨迹是否仍可作为 manipulation 训练数据，必须结合轨迹完整性、跌倒、tracking、contact、物体终态以及任务专属语义判定；最后仍应通过 replay 或自动 evaluator 确认。

---

## 2. 研究问题是怎样逐步明确的

### 2.1 起点：换 checkpoint 后再看 CSV

最早的设想是：

```text
从原 data.csv 构造 G1 encoder 输入
→ 运行 low-latency encoder/decoder
→ 用输出替换 data.csv 的 qpos
→ 在 MuJoCo replay 中判断任务是否完成
```

讨论后发现，这个思路混淆了三种不同的量：

```text
raw_action
  → default pose / scale / clip / joint reorder
q_target（或 q_cmd）
  → PD + torque limit + gravity + inertia + contact
实际 qpos
```

因此不能把 decoder 输出直接写进 qpos。尤其是拿取、放入抽屉、踩垃圾桶等接触任务，如果没有重新运行物理闭环，任务物体的运动、接触力和机器人 root 都会变成伪造状态。

### 2.2 第一版可执行问题

问题随后改写为：

> 在原 recording 的场景、任务物体初态、手指目标和 reference 时间轴下，让另一个 checkpoint 真正闭环控制 MuJoCo，保存新的 qpos 和物体轨迹，再判断任务是否仍成功。

这形成了 `change_ckpt/`。

### 2.3 第二个问题：reference intent 还是旧执行轨迹

随后提出另一种比较：把原 recording 的实际 qpos 抽出来，作为 tracker 的 reference，再分别运行 regular、low-latency 或其他 controller。

这形成了 `change_ckpt_track/`，也促使我们明确：

| 模式 | 输入含义 | 回答的问题 |
|---|---|---|
| `reference_motion` | 原采集系统发给 SONIC 的动作意图 | 换 controller 后能否执行同一上游意图 |
| `executed_qpos` / qpos-track | 旧 controller 已经执行出的实际姿态 | 新 controller 能否模仿旧执行轨迹 |

二者都值得比较，但不能混成同一个实验。

### 2.4 最终研究问题

随着 Teleopit 和 SONIC v1.1 被纳入，最终问题变成：

> 对一段已经成功的接触型 humanoid loco-manipulation recording，在保持任务场景、动作意图和手部目标不变的条件下，更换独立训练的全身 controller，任务是否仍然成功？如果失败，失败来自 reference 表达、历史状态、root 漂移、低层 PD、时序还是接触动力学中的哪一部分？生成的新轨迹还能否作为 VLA 训练数据候选？

---

## 3. 对话与实现的阶段时间线

| 阶段 | 主要内容 | 形成的结果 |
|---|---|---|
| A | MuJoCo 3.2/3.10、旧 replay、资源路径、launch 环境 | `.venv_sim` 与 `.venv_replay` 分离；确认 replay 只按 qpos 显示 |
| B | 解析 `data.csv`、验证 640 reference、encoder token、decoder history | 建立 CSV 字段语义和 SONIC 输入结构 |
| C | 用原 `reference_motion` 闭环换 regular/low-latency | `change_ckpt/` |
| D | 用原实际 qpos 构造 50 Hz tracker reference | `change_ckpt_track/` |
| E | ghost compare、root assist、200/400 Hz 与 viewer timing | 比较播放器、root-assist ablation、时序诊断 |
| F | source-history prefill 与 policy offset 统一 | 去除零历史启动偏差，明确 raw group 11 |
| G | 接入 SONIC v1.1 | regular / low-latency / v1.1 对照 |
| H | 接入 Teleopit，研究跨 controller reference adapter | `Teleopit_rollout/` |
| I | 统一确定性实验协议 | `controller_replacement/` |
| J | 替换为 NVIDIA 官方 regular 模型/config，清理 planner | 官方 1762 维 regular 接口 |
| K | 实时 reference ghost 与 staged motion 工具 | `sim_reference_overlay/` |

---

## 4. 环境、旧 replay 与 MuJoCo 资源

### 4.1 `.venv_sim`、`.venv_replay` 与 IsaacLab

对话中曾临时把 `.venv_sim` 的 MuJoCo 从 3.10 改为 3.2，以满足旧二进制 `libmujoco.so.3.2.0`；后来按要求恢复：

- **[已验证]** `.venv_sim`：MuJoCo 3.10.0，用于官方 sim、`change_ckpt`、`change_ckpt_track` 和 overlay；
- **[已验证]** `.venv_replay`：MuJoCo 3.2.0，用于兼容旧 replay/compare；
- **[已验证]** `Teleopit_rollout/.venv`：用于 Teleopit 和当前统一 `controller_replacement` rollout；
- IsaacLab Conda 环境：用于依赖 IsaacLab 的训练、评估和旧 PyTorch checkpoint 重建。

一个 native ELF 并不真正“属于”某个 Python venv。shell prompt 可以显示 `(isaaclab)`，但旧 replay 最终加载哪一份 MuJoCo，取决于动态链接器能否在 `LD_LIBRARY_PATH`、rpath 或系统目录中找到 `libmujoco.so.3.2.0`。

### 4.2 `launch.json` 的环境判断

早期检查表明，`train_agent_trl` 和 `eval_agent_trl` 的 VS Code debug 配置没有显式绑定 `.venv_sim`，因此会使用 VS Code 当前选择的 Python interpreter；这类脚本依赖 IsaacLab，表现为在 IsaacLab Conda 环境中运行是合理的。`.venv_sim` 的主要用途是独立 MuJoCo simulator，而不是 IsaacLab 训练。

### 4.3 旧 C++ replay 实际做什么

**[已验证]** `sample_data/ztj/replay_mujoco_csv.cpp` 的核心流程是：

1. 读取 recording 的 `data.csv`；
2. 识别 `qpos[...]` 列；
3. 按行把 qpos 写入 `mjData`；
4. 调用 `mj_forward` 更新几何位置；
5. 按记录时间显示。

它不会重新运行 policy，也不会通过 `mj_step` 重新积分动力学。虽然能够识别 qvel，原 replay 的画面主要由 qpos 决定。因此：

- 身体姿态来自 qpos；
- 手指姿态也来自 qpos；
- 抽屉、扫描物、垃圾桶踏板/桶盖等可动物体若属于场景 qpos，也由相同行的 qpos 恢复；
- token、reference、raw action、手目标等日志列不是原 replay 显示所必需的。

### 4.4 资产缺失错误

错误：

```text
Error opening file '.../model_snapshot/mujoco/model/g1/pelvis.STL'
```

说明 snapshot 中的 XML 被找到了，但其相对引用的 G1 mesh 没有处于预期的 `model_snapshot/mujoco/model/g1/` 布局。仅放入 `task_assets` 不能替代机器人基础 mesh；snapshot、机器人模型资产和任务资产都需要保持 XML 所期待的相对路径或有效链接。

### 4.5 replay 的用途边界

旧 replay 很适合回答：

- 原 recording 的 qpos 实际是什么；
- 新 rollout 保存的 qpos 看起来如何；
- 手和任务物体终态有没有变化。

它不能回答：

- 当前 checkpoint 是否能在线稳定控制；
- raw action 经过 PD 后会产生什么接触；
- 在另一个 controller 下任务是否真实成功。

---

## 5. 原始 `data.csv` 的结构与字段语义

### 5.1 三份数据格式比较

对以下三份 recording 做过列名和尺寸比较：

| Recording | 行数 | 列数 | qpos | qvel | 场景可动物体示例 |
|---|---:|---:|---:|---:|---|
| `20260612_144214_g1_sim` | 2035 | 1564 | 52 | 51 | 垃圾桶踏板、桶盖 |
| `20260720_144342_g1_sim` | 4591 | 1579 | 60 | 58 | 三个抽屉、scanner/free joint |
| `20260722_145020_g1_sim` | 3247 | 1573 | 57 | 55 | can/free joint |

**[已验证]** 它们并不是整张表每一列都相同，因为场景 qpos/qvel 数量不同；但有 1560 个共同列名，从 `policy_valid` 开始的 1457 个控制/策略字段名字一致。三份数据都以约 0.0025 s 间隔记录，policy 有效尺寸一致：token 64、reference 640。

### 5.2 顶层布局

典型布局如下：

1. 场景与时间：
   - `scene_path`
   - `sample_index`
   - `control_time_s`
   - `mujoco_time_s`
2. `qpos[...]`
3. `qvel[...]`
4. policy 元数据：
   - `policy_valid`
   - `policy_seq`
   - `policy_token_size`
   - `policy_reference_motion_size`
5. `token_state[0:256]`
6. `reference_motion[0:1024]`
7. `policy_last_action_in[0:29]`
8. `policy_raw_action_out[0:29]`
9. `policy_received_dof_pos[0:43]`
10. 左右手的 q、dq、kp、kd、tau 和 close 标记。

### 5.3 `qpos` 与 `qvel`

`qpos` 是 MuJoCo 的广义位置，不等于“机器人关节角”这一小部分。它通常包括：

- G1 floating base 的 xyz 和 quaternion；
- 29 个身体关节；
- 左右手 14 个关节；
- 当前任务场景中的抽屉、物体 free joint、踏板或桶盖等自由度。

因此原 CSV **确实包含 root 位置**，也能够保存任务物体实际运动。

`qvel` 是对应的广义速度。free joint 的 qpos 为 7 维，而 qvel 为 6 维，所以 qpos 与 qvel 总数不一定相同。

### 5.4 `policy_seq` 与“去重”

原 logger 约为 400 Hz，而 policy inference 约为 50 Hz，因此同一个 `policy_seq` 通常连续出现约 8 行。

这里所谓按 `policy_seq` “去重”并不是删除重复的物理状态，而是：

> 从 400 Hz 日志中，为每次 50 Hz policy inference 选择一个代表行，得到 policy-frame 序列。

同组的 8 行 qpos 仍可能继续变化；只是 token/reference/raw action 等 policy metadata 通常保持同一推理结果。

### 5.5 `reference_motion[0:1024]`

CSV 为兼容性预留 1024 列，但 regular SONIC 的有效 reference 是前 640 维：

```text
10 × 29 joint position       = 290
10 × 29 joint velocity       = 290
10 × 6 relative orientation  =  60
                                   ───
                                   640
```

这不是仅从列名猜出的。验证依据包括：

- observation config 和 encoder 构造代码；
- 每个 rot6d slot 的旋转矩阵正交性质；
- 相邻 policy frame 间的 slot 重合关系；
- 将 640 维按 regular encoder 布局送入原 encoder 后，64 维 token 与 CSV 完全一致。

### 5.6 anchor orientation 的真实含义

早期曾出现“是不是直接输入 reference orientation”的疑问。正确关系是：

```text
q_relative = inverse(q_robot_anchor) * q_reference_anchor
q_relative → rotation matrix → 前两列，即 6D rotation
```

所以 CSV 最后 60 维不是简单的世界系 absolute quaternion。

如果要让新 rollout 使用相同世界系 reference orientation，需要：

1. 结合原 recording 对应的 robot quaternion，把存储的 relative orientation 还原成 reference world quaternion；实际 adapter 必须遵守原 encoder 的 base-sample 约定，当前 regular 恢复使用 `previous-index5`，不应想当然地取 CSV 同行 pelvis quaternion；
2. 再结合新 rollout 当前 robot orientation，重新计算新的 relative orientation。

即使某几个 future slot 的 reference world pose 完全重复，只要新 robot 自身姿态变化，下一次 policy inference 得到的 relative orientation 仍会改变。

不过，同一 `policy_seq` 对应的约 8 个 400 Hz CSV 行通常重复保存同一次 50 Hz 推理的 reference，不会在每个 logger tick 重新写一份 orientation。

### 5.7 token、raw action 与 received state

- `token_state[0:64]`：SONIC encoder 输出的有效 token；容量到 256；
- `policy_raw_action_out[0:29]`：decoder 输出的 29 维身体 raw action；
- `policy_last_action_in[0:29]`：进入当前 decoder history 的 previous action；
- `policy_received_dof_pos[0:43]`：policy 实际收到的身体 29 + 双手 14 的 measured joint state 快照。

`policy_received_dof_pos` 不是目标值。由于旧采集链异步，它和同一 CSV logger 行的 qpos 可能存在相位差；这也是后来 takeover 初始化需要做 phase match 的原因。

### 5.8 手部字段

**[已验证]** SONIC body decoder 只输出 29 个身体关节，不控制手。手目标来自独立链路。

手部字段呈现典型命令结构：

```text
q target, dq target, kp, kd, tau feed-forward
```

实际应用关系可概括为：

\[
\tau_{\rm applied}
=
\tau_{\rm ff}
+k_p(q_{\rm cmd}-q_{\rm pos})
+k_d(\dot q_{\rm cmd}-\dot q_{\rm pos})
\]

这是一个公式，不是两个公式。

原 replay 中手指动作由 qpos 中的实际手关节决定；后面的手部字段用于记录遥操目标、增益和命令链。它们对离线显示不是必需的，但对重新闭环 rollout 非常重要。

`left_hand_close/right_hand_close` 更像上层二值意图。由于实际数据采集脚本不在仓库中，PICO trigger/grip 到 7 个手指 target 的精确映射仍未验证。

---

## 6. SONIC encoder 与 decoder 的复现验证

### 6.1 regular encoder 的精确 token 验证

使用 `20260720_144342_g1_sim` 做过最严格的一次检查：

- 按每个唯一 `policy_seq` 取一帧，共 575 帧；
- 从 CSV 重建 regular G1 encoder 的完整输入；
- 分别比较 PyTorch encoder、导出的 ONNX encoder 与 CSV `token_state[0:64]`。

**[已验证]** 三组比较均为：

```text
575 / 575 frames
max_abs = 0
```

这证明当时使用的旧 regular G1 encoder 的 reference 切分、history reshape、MLP、FSQ 和 CSV policy 行对齐能够被精确恢复。

后来替换为 NVIDIA 官方 regular release 模型后，官方 encoder 输入为 1762 维；适配器修正后也完成固定输入 token 精确验证。不能把旧 1751 维 wrapper 和当前官方 1762 维接口混为一谈。

### 6.2 decoder 的 994 维输入

SONIC decoder 输入结构为：

```text
当前 token 64
+ 10 × history 93
= 994
```

每个 93 维历史状态为：

| 分量 | 维度 |
|---|---:|
| angular velocity | 3 |
| body joint position | 29 |
| body joint velocity | 29 |
| previous raw action | 29 |
| projected gravity | 3 |
| 合计 | 93 |

训练中可能使用 observation noise 或 corruption，但部署推理并不会因此随机加噪。早期 decoder 不能逐值复现 source raw action，不能简单归因于 noise；更明确的来源是：

- 新进程 history 初始为空；
- received state 与 logger 同行 qpos 的相位不完全一致；
- previous action 恢复方式；
- 少量 current token 量化 bin 差异；
- 原 solver、contact、旧 ctrl、qacc warm-start 和通信相位没有保存在 CSV 中。

### 6.3 raw action 到实际 qpos

raw action 通常先经过：

```text
joint reorder
clip
q_target = default_joint_angle + action_scale × raw_action
```

随后通过 controller 原生 PD 和 MuJoCo 得到实际 qpos。因此：

- raw action 不是关节角；
- q_target 不是实际关节角；
- qpos 才是物理执行结果；
- 原数据只保存 qpos 仍不足以还原全部闭环内部状态。

---

## 7. `run_sim_loop.py` 与原始 sim2sim 能证明什么

用户原先的 sim2sim 使用：

```bash
# Terminal 1：MuJoCo
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2：regular
cd gear_sonic_deploy
bash deploy.sh sim

# Terminal 2：low-latency
./deploy.sh \
  --cp policy/low_latency/model \
  --obs-config policy/low_latency/observation_config.yaml \
  sim
```

`gear_sonic/scripts/run_sim_loop.py` 本身只是 simulator 入口；核心循环在 simulator 基类中：

- 接收 DDS command；
- 读取当前 MuJoCo state；
- 计算 PD；
- `mj_step`；
- 发布 LowState；
- 可选 viewer。

**[已验证]** 上述 sim2sim 确实运行了 checkpoint 的 encoder、decoder、action mapping、PD 和 MuJoCo 闭环，所以它能做 checkpoint 加载、站立、默认动作跟踪和稳定性冒烟测试。

但当时显示的是项目自带 reference 序列，并不来自任务 recording 的 `data.csv`。默认参考与 `gear_sonic_deploy/reference/example` 一类资源有关；`sample_data/robot_filtered` 更接近训练 motion library，而不是这次任务 replay 的直接输入。

因此 sim2sim 不能直接回答“换 checkpoint 后原抽屉/垃圾桶任务是否完成”，但它说明我们不必重新发明 SONIC 的 action mapping、PD 和部署逻辑，并为后续 wrapper 提供了可信基线。

---

## 8. `change_ckpt/`：保留原 `reference_motion`

### 8.1 设计目标

`change_ckpt/` 是第一套真正针对原任务 recording 的闭环替换方案：

```text
原 data.csv 中 reference_motion + 手目标
→ 50 Hz CSV/ZMQ publisher
→ C++ deploy 在本地运行 encoder + decoder
→ DDS 关节命令
→ MuJoCo 闭环
→ 新 data.csv
```

重要设计：

- 不发送原 token，待测 checkpoint 必须重新编码；
- 不把原 qpos 逐帧覆盖回机器人；
- 原 qpos/qvel用于 takeover 初态、场景物体初态和时间对齐；
- 任务物体随后由新接触动力学真实演化；
- 原手目标通过独立手 PD 链恢复；
- 原项目代码不直接修改，新增内容放在 `change_ckpt/`；
- 输出写入 `change_ckpt/data/<recording>_<checkpoint>`，重复同条件覆盖。

旧 DDS 链无法把 C++ deploy 内部的新 token/raw action 直接回写到 simulator 侧 replay `data.csv`。因此该文件中的 `policy_valid` 为 0，token/raw-action 等 policy 列是零占位；真正的 SONIC 内部 token、action、q、dq 等应查看同一结果的 `deploy_csv/`。这与后来统一程序直接在 `data.csv` 写入 SONIC 真值的行为不同。

### 8.2 regular 与 low-latency

- regular：保留 recording 中已经保存的十个 slot；
- low-latency：按其 observation config 使用连续短窗口；
- 两者使用各自匹配的 encoder、decoder 和 config。

原 sim2sim 已经证明两套模型能运行；`change_ckpt` 进一步检验同一任务 reference 下的接触表现。

### 8.3 已知任务定义

由用户明确确认：

- `20260720_144342_g1_sim`：拿起物体、放入抽屉、合上抽屉；
- `20260612_144127_g1_sim`：踩上垃圾桶后保持稳定。

其余 recording 若没有明确语义，只应报告轨迹、物体位姿和接触变化，不能根据文件名猜任务。

### 8.4 旧异步链的时序

默认旧模式：

| 环节 | 频率 |
|---|---:|
| source CSV logger | 400 Hz |
| simulator/DDS control | 200 Hz |
| policy/reference | 50 Hz |
| MuJoCo physics | 1 kHz |
| viewer | 50 Hz |

因此每个 50 Hz policy action 约保持 4 个 200 Hz control tick，而每个 control tick 消费两行 400 Hz source。

曾增加 400 Hz control / 2 kHz physics 诊断模式，但它会把每个 control tick 的墙上时间预算压到 2.5 ms。加入 root alignment 后的 `mj_forward`、DDS、日志和 viewer 很容易 overrun。**原 CSV 为 400 Hz 并不等于原 controller 也是 400 Hz，也不等于改成 400 Hz 一定更准确。**

### 8.5 viewer 与 shell 批量结果不同

用户曾发现：shell/no-viewer 生成的 regular 可以踩到垃圾桶，而手动 viewer 运行不能；`run_all.sh` 还会在第一个文件夹后停止。

结论是：

- shell 循环本身不会改变 checkpoint；
- 旧异步链依赖 DDS 最新值、ZMQ reference 和墙上时间；
- viewer 负载、端口残留、进程退出状态或某一步返回非零都可能改变相位或触发 `set -e` 停止；
- 正式旧链实验应 no-viewer，并检查 `wall_clock_timing_valid`、RTF、deadline rebase 和 hand overrun；
- 后来的统一程序改为固定仿真步，正是为了消除这类机器负载依赖。

### 8.6 旧链中一组有代表性的物理结果

`change_ckpt` 文档保留了一组通过 headless 实时质量门的 regular/low-latency 对照：

- regular：scanner 位移约 0.734 m，抽屉进度指标约 103.8%；
- low-latency：scanner 位移约 0.020 m，抽屉进度指标约 53.9%。

它说明两套 checkpoint 都能稳定跑完，并不等于两者都完成了完整任务。特别是 low-latency 的物体运动不足以支持“拿起、放入并关闭抽屉”的语义成功；最终仍需 replay 或 task evaluator。

---

## 9. `change_ckpt_track/`：用实际 qpos 构造 50 Hz reference

### 9.1 工作方式

`change_ckpt_track/` 从原 recording 的实际执行轨迹构造 tracker 目标：

1. 按 `policy_seq` 得到 50 Hz policy frames；
2. 提取原实际 pelvis quaternion、29 个身体关节和速度；root xyz 另存为初始化、assist、导出和比较信息；
3. 构造 controller 所需的 reference slots；
4. 恢复原场景、任务物体初态和手目标；
5. 仍通过 C++ SONIC deploy 和 MuJoCo 闭环运行。

它测试的是“新 controller 能否跟踪旧 controller 已执行的轨迹”，而不是“能否执行原上游动作意图”。

需要特别注意：protocol v1 发给 SONIC C++ encoder 的是 body q/dq、绝对 reference quaternion 和手目标，**不会把 root xyz 作为 regular/low-latency encoder 输入**。原 root xyz 只参与仿真初始化、可选 root assist 和结果比较。

### 9.2 future slot 的 nominal 与 observed

regular config 的名义 future offset 是：

```text
[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
```

以 50 Hz policy 计，对应最长约 0.9 s future horizon。

但对原 CSV 数值匹配后，发现实际保存的 slot 经常类似：

- `20260720_144342` 一类：`[0,5,9,9,...,9]`；
- 抽查的若干 `20260612`：`[0,4,4,4,...,4]`；
- `20260722_154958`：约 `[0,5,8,8,...,8]`。

这里的 `future_slot_policy_lags` 是分析结果：把当前 policy 的每个 recorded slot q/dq，与后续 policy frame 的 `reference_motion` **slot 0** 比较，寻找数值对应项，再计算后续 frame 与当前 `policy_seq` 的差。它不是 CSV 原生列，也不是拿 slot 去匹配后续实际 qpos。

**[已验证]** 多份数据从采集开始就存在尾部 slot 重复，并非只有动作结束时 hold。

**[待解决]** 没有原始采集脚本，因此无法最终断言这是 publisher lookahead 不足、末帧保持、数据窗口构造错误还是别的机制。训练中的 `freeze_frame_aug` 也不能直接解释采集 CSV 的模式。

### 9.3 为什么 qpos-track 可能比 reference-motion 差

用户通过 replay 指出，qpos-track 的主要问题是脚和手 tracking 较差，而不是简单“接触位置有了但没有力”。更合理的解释包括：

- 实际 qpos 已经包含旧 controller 的 tracking error 和接触反馈；
- 把它重新当作动作意图会二次编码旧误差；
- 原 reference 的目标 dq 和 future phase 不能从一个当前 qpos 完整恢复；
- canonical `[0,5,...,45]` 与原 recorded 重复 slot 的 lookahead 不同；
- anchor orientation 会结合新 robot 状态重新计算；
- 脚部几个时刻的小相位差会改变接触，继而造成 root 和手部整体偏差。

因此 `executed_qpos` 适合作为 controller-following ablation，但不应替代正式的 `reference_motion`。

---

## 10. ghost compare、播放速度和 root assist

### 10.1 compare viewer 的视觉语义

比较播放器后来统一为：

- 实心机器人：新 rollout；
- 青色半透明 body：原 recording 同一时刻的实际 qpos；
- 橙色半透明手：原 recording 同一时刻的实际手指 qpos；
- `--ghost-mode source`：默认，直接比较新轨迹与原成功执行轨迹；
- `--ghost-mode reference`：显示构造给 tracker 的 50 Hz reference，适合检查 tracker 跟踪误差。

曾出现橙色 ghost 手指异常巨大，后来定位为 ghost joint/geom 映射问题并修复。

### 10.2 为什么 Python compare 看起来更慢

Python compare 每个显示帧需要：

- 更新实心机器人；
- 更新 ghost；
- 对两个状态做 `mj_forward`；
- 同步 viewer。

**[推测]** 如果它试图逐个显示 200/400 Hz 数据行，双机器人状态更新、`mj_forward` 和 viewer sync 的额外负载可能让墙上播放低于实时。旧 C++ replay 更接近“数据线程按时间推进，画面取最新状态”，因此可能看起来更快。没有专门 profiling 时，不能断言瓶颈一定就是两个 `mj_forward`；可以确定的是播放观感差异不等于保存轨迹的仿真时间真的变慢。

### 10.3 root assist 的实现

运行时 `root-assist xy` 的语义是：

```text
在每个 200 Hz PD 边界：
robot root qpos[x,y] ← 同仿真时刻的 source root[x,y]
robot root qvel[vx,vy] ← 同仿真时刻的 source root velocity[x,y]
```

它不覆盖 root orientation、身体关节、手指或任务物体。旧实验还试过 `xyz`，额外覆盖 z/vz；统一正式协议只保留 `none` 和 `xy`。

如果前一帧已经有明显误差，硬覆盖会产生相应跳变；高频时误差通常较小，所以单次跳变也较小，但本质仍是 teleport/oracle。

把 assist 放在运行时而不是 replay 的好处是：下一次 policy 会看到修正后的 robot state，脚和物体接触也按修正位置真实计算。缺点是它不再测量 controller 自身的全局 root 能力。

### 10.4 用户观察到的效果

- **[用户观察]** old `change_ckpt` 中，low-latency no-viewer + root assist xy 后能够完成此前失败的任务；
- **[用户观察]** `20260612_144127` 的 xy 仍未踩到垃圾桶，而 xyz 可以踩到，但与原轨迹仍有差异；
- **[结论]** assist 结果必须和 no-assist 分开报告，xyz 更只能作为接触高度的 oracle 诊断。

### 10.5 root drift 由什么决定

SONIC regular reference 中没有绝对 root xy 误差反馈，因此：

- 第一帧 root 对齐只保证起点；
- 脚 tracking、支撑相位、速度偏差和接触冲量会逐步积分成 root drift；
- decoder history 初始化会改变最初几步 action；
- PD、torque limit、MuJoCo solver 和时序也会改变落脚；
- 不同 recording 对这些扰动的敏感程度不同。

这解释了为什么同样 regular/no-assist，有的 recording 几乎不漂，有的很快漂移；不能只用“第一帧是否对齐”解释。

---

## 11. source-history prefill、takeover 与 offset

### 11.1 为什么从第十帧开始仍可能是零历史

把 simulator 和 reference publisher 从 recording 后面的某个 frame 开始，并不会自动填充新 C++ deploy 的 `StateLogger`。新进程内部仍然没有过去十帧，因此 decoder 仍会收到 padding。

source-history prefill 的做法是：

1. 从 takeover 前恢复 9 个 source policy state/action；
2. 用当前 live state 组成第 10 帧；
3. 第一次新 controller inference 就使用完整 10 帧 history。

### 11.2 严格 A/B 结果

对 `20260612_144154`，在相同 scene、initial state、reference、wrapper 和 checkpoint 下，只切换 zero history 与 source-history prefill：

| 指标 | zero history | source-history prefill |
|---|---:|---:|
| first action RMSE vs source | 0.817313 | 0.051266 |
| 前 0.2 s body RMSE | 0.13311 rad | 0.00531 rad |
| 前 0.2 s root XY RMS | 1.300 cm | 0.085 cm |
| 前 1 s body RMSE | 0.07146 rad | 0.00867 rad |
| active root XY RMS | 11.97 cm | 4.48 cm |

first action 误差下降约 93.7%。垃圾桶踏板/桶盖也从几乎不动变为发生明确运动，但末尾仍未完全复现原任务状态。

这说明零历史是重要误差源，但 prefill 不能恢复：

- 原 solver warm-start；
- 原 ctrl；
- 完整 contact state；
- DDS/ZMQ 相位；
- 旧 controller 此后真正会产生的 action。

完整 A/B 记录见 [`change_ckpt/SOURCE_HISTORY_PREFILL_RESULTS.md`](change_ckpt/SOURCE_HISTORY_PREFILL_RESULTS.md)。七份 qpos-track regular 的 prefill 对照、raw/processed offset 对应关系见 [`change_ckpt_track/SOURCE_HISTORY_PREFILL_RESULTS.md`](change_ckpt_track/SOURCE_HISTORY_PREFILL_RESULTS.md)。

### 11.3 source-history 与 self-warmup 不是同一件事

- **source-history prefill**：takeover 前的历史由旧 source controller 提供，让第一次推理不从全零开始；
- **self-warmup**：新 controller 接管后的前 10 次推理，逐步把 history 中旧 action 替换成自己的 action。

最终约定：

- 前 10 次新 action 全部真实作用于机器人和物体；
- 这些数据保留在完整轨迹中；
- 从第 11 次新 inference 开始，history 才完全由新 controller 自己产生；
- tracking 汇总从第 11 次开始；
- 任务成功判定仍必须包含 warmup 对物体造成的真实后果。

### 11.4 offset 最终定义

早期曾混用 processed offset 10/11 与 raw policy offset，尤其当第一组 `policy_seq` 不完整时容易产生歧义。

最终统一：

- 原 CSV 第二个 raw `policy_seq` group 定义为公开 offset 0；
- 旧两个 launcher 默认公开 offset 10；
- 其实际等价于 zero-based raw group 11，即第 12 个 raw group；
- `controller_replacement` 直接使用清晰参数 `--raw-policy-group-offset 11`；
- 第一组是否完整不再改变真实 takeover group；
- source-history prefill 为默认；
- 新输出目录不再把 `start_offset_*_source_history_prefill` 写进名字。

---

## 12. SONIC v1.1

### 12.1 模型接口

**[已验证]** 当前 v1.1：

- encoder：1751 → 64；
- decoder：994 → 29；
- 模型和 config 的 SHA-256 与下载来源对应；
- 已接入 `change_ckpt`、`change_ckpt_track` 和统一方案。

### 12.2 heading normalization

regular 的方向特征可概括为：

\[
o_{regular}=R6D\left(q_{robot}^{-1}q_{ref}\right)
\]

v1.1 先从当前机器人姿态提取 heading/yaw frame：

\[
o_{v1.1}=R6D\left(H(q_{robot})^{-1}q_{ref}\right)
\]

这里的 `o` 是 encoder 的 orientation observation，不是 action，也不是一个让 root 移动的命令。

“heading normalization”也不能简化为“v1.1 只关心 yaw 差异”：

- 它改变的是 reference orientation 的表达坐标系；
- reference 自身 roll/pitch 信息并非完全删除；
- robot 当前 roll/pitch 仍能通过 gravity、angular velocity 和 decoder history 进入控制；
- 它不提供绝对 root xy tracking，因此不能自动解决平移漂移。

### 12.3 在 `reference_motion` 路径中的转换

不能把旧 regular reference 最后 60 维原样送入 v1.1。当前做法是：

1. 保留前 580 维 q/dq；
2. 用 source robot pelvis quaternion 与旧 relative rot6d 恢复十个世界系 reference quaternion；
3. v1.1 rollout 时再结合新 robot 当前 heading，实时生成新的 60 维 heading-normalized orientation。

这保留了原 recording 实际 slot 内容，包括 `[0,5,9,9,...]` 一类重复，而不是强制重建理想窗口。

### 12.4 已有结果的正确解释

- qpos-track 中 v1.1 多数可稳定跑完整段，body RMSE 常位于 regular 与 low-latency 之间；
- `20260720` 的 scanner 终点在 qpos-track 中没有复现原放置；
- reference-motion + root assist 的部分物体终态接近 source，但若 timing validity 失败，不能纳入正式结论；
- 对没有明确任务语义的数据，只报告轨迹和物体状态，不猜测“成功”。

详见：

- [`change_ckpt/SONIC_V1_1_REFERENCE_RESULTS.md`](change_ckpt/SONIC_V1_1_REFERENCE_RESULTS.md)
- [`change_ckpt_track/SONIC_V1_1_RESULTS.md`](change_ckpt_track/SONIC_V1_1_RESULTS.md)

---

## 13. Teleopit：替换为非 SONIC controller

### 13.1 为什么它是有价值的基线

只比较 regular、low-latency 和 v1.1，仍属于同一 SONIC 家族。Teleopit 提供了一个独立 observation、历史网络、action mapping 和 PD 的 tracker，可以更直接检验“接触型任务数据能否跨 controller 使用”。

### 13.2 Teleopit v0.5 输入

固定使用的 `track_g1.onnx`：

- 当前 observation：167 维；
- history：10 × 167，包含当前帧和过去 9 帧；
- 输出：29 维 body raw action。

167 维布局：

| 分量 | 维度 |
|---|---:|
| reference q | 29 |
| reference dq | 29 |
| torso relative rot6d | 6 |
| robot base angular velocity | 3 |
| robot q - default q | 29 |
| robot dq | 29 |
| previous raw action | 29 |
| robot projected gravity | 3 |
| reference torso linear velocity | 3 |
| reference torso angular velocity | 3 |
| reference gravity | 3 |
| reference torso height | 1 |
| 合计 | 167 |

这十帧 history 是过去约 0.18 s 的 observation，不是 SONIC regular 的十个未来目标。

### 13.3 tracker 网络结构与训练信息

基于 pinned Teleopit v0.5 代码与 ONNX 检查，网络本身可确认：

- history TemporalCNN：`167 → 256 → 128 → 64`，kernel 3、padding 1、ELU、global average pooling；
- 将 64 维 history embedding 与当前 167 维 observation 拼成 231 维；
- actor MLP：`231 → 2048 → 1024 → 512 → 256 → 128 → 29`；
- deterministic actor 参数量约 3,517,661；
- actor 使用上述确定性 ONNX 推理结构。

仓库公开的训练配置另显示 asymmetric critic，以及 PPO 的 clip 0.2、5 epochs、4 minibatches、lr 5e-4 adaptive、desired KL 0.01、gamma 0.99、lambda 0.95、rollout length 24。**这些是仓库配置，不等价于发布的 `track_g1.onnx` 已由独立 run manifest 证明逐项采用了同一参数。** 文档/配置还出现 30k 与 checkpoint iteration 40k 等不同训练长度线索，不能据此反推唯一训练过程。

当时对公开仓库和文档的检查没有找到足以唯一复现该发布 checkpoint 的完整 motion 数据集清单、采样权重和训练 run manifest。因此“网络结构/输入输出已验证”和“发布模型究竟用哪些 motion、多少样本训练”必须分开；后者仍是待确认项。

这部分记录的是我们使用的 Teleopit tracker，不再把 OASIS 论文中可能使用的不同版本 tracker 参数混入结论。

### 13.4 torso link、pelvis 与 root

Teleopit 的 reference anchor 是 `torso_link`，而不是 SONIC 的 pelvis。torso 是 pelvis 上方通过腰关节连接的刚体，二者姿态和速度并不相同。

虽然 167 维 observation 没有 absolute reference root x/y 字段，构造 reference 时仍需 root 信息：

- 相邻 root xyz 参与 reference torso linear velocity；
- root z 参与 torso height；
- pelvis quaternion + body joints 经 FK 得到 torso orientation；
- reference gravity和角速度也来自 torso/pelvis 运动学。

因此 Teleopit 仍可能积累 world-frame xy drift；“输入中没有 root xy”不代表它会自动跟踪绝对位置。

### 13.5 手部为什么不需要 Teleopit Dex3 checkpoint

SONIC 本身也不输出手。既然实验目标是替换 body whole-body controller，而手部仍由原 recording 的独立目标链控制，那么 Teleopit 只需要 29-DoF body tracker；双手继续使用相同 source target 和手 PD。只有研究问题变成“连手 controller 也一起替换”，才需要额外 Dex3 checkpoint。

### 13.6 两种 Teleopit reference

第一版 `qpos-track`：

```text
recorded actual root xyz/quaternion + actual body qpos
→ FK
→ Teleopit torso reference
```

后来增加 `sonic_reference_hybrid`：

```text
SONIC reference_motion slot 0 的 29 body q
+ 从 relative orientation 恢复的 reference pelvis world quaternion
+ recording actual root xyz
→ Teleopit FK
→ torso reference features
```

它必须叫 hybrid，因为 SONIC 640 维 reference 不包含独立的 reference root xyz，只能借用 recording 的实际 root translation。

只用 slot 0，是因为 slots 1–9 是 SONIC future targets；它们不是 Teleopit 的 past observation history。

### 13.7 已有 Teleopit 结果

- **[已验证数值]** `20260720 + xy` 的 hybrid scanner 最终位置误差约 2.25 cm，相比 qpos-track 明显改善；但“是否放入并关闭抽屉”仍需完整 replay；
- `20260612_144127` 即使使用旧 xyz assist，pedal/lid 也没有复现原任务运动；
- `20260722_154958 + xy` 的物体位置接近 source，但姿态仍有差异；
- no-assist 下部分轨迹存在明显 root drift 或意外接触。

这支持“SONIC reference intent 可以被适配为 Teleopit 输入”，但不支持“更换 controller 后所有任务都会自然成功”。详见 [`Teleopit_rollout/SONIC_REFERENCE_HYBRID_RESULTS.md`](Teleopit_rollout/SONIC_REFERENCE_HYBRID_RESULTS.md)。

### 13.8 四套实现的直接对照

| 实现 | 主 reference | Controller | 执行链 | 固定时钟/墙钟 | 主要用途 |
|---|---|---|---|---|---|
| `change_ckpt` | 原 `reference_motion` | SONIC regular/low-latency/v1.1 | Python publisher → ZMQ → C++ TensorRT → DDS → MuJoCo | 旧异步墙钟；默认 1 kHz physics/200 Hz control/50 Hz policy | 最接近原 SONIC deploy 的任务 reference 替换 |
| `change_ckpt_track` | 原实际 qpos 构造的 50 Hz reference | SONIC family | 与 `change_ckpt` 相同 | 同一旧异步链 | executed-qpos/qpos-track ablation |
| `Teleopit_rollout` | actual qpos 或 SONIC hybrid | Teleopit v0.5 | Python ORT 单进程 | 2 kHz physics/200 Hz PD/50 Hz policy/400 Hz CSV | 验证非 SONIC controller 可行性 |
| `controller_replacement` | `reference_motion` 主实验；`executed_qpos` ablation | 四种 controller | Python ORT 单进程，无 DDS/ZMQ | 全部由整数 simulation step 决定 | 当前正式统一比较协议 |

旧 `change_ckpt`/track 的 `data.csv` 中 token/raw action 可能是 simulator 侧零占位，真实 SONIC 内部输出需查看 `deploy_csv/`；统一程序则把 SONIC 真值写回兼容列，并把各 controller 的原生数据放入 telemetry。

---

## 14. 统一正式实现：`controller_replacement/`

### 14.1 为什么重新实现

旧三条链各自证明了很多问题，但也留下差异：

- `change_ckpt`/`change_ckpt_track` 使用 C++ TensorRT + DDS/ZMQ + 墙上时间；
- Teleopit 使用 Python/ORT 单进程；
- viewer 会影响旧异步时序；
- 输出字段、history、metrics 和 provenance 不统一；
- 各 controller 的 angular velocity、PD 和 torque limit 容易被错误统一。

因此新建 `controller_replacement/`，保留旧实现用于回归，不直接修改它们。

### 14.2 用户确定的正式协议

**[用户决策]**

- 暂不做异步 C++ deploy streaming 鲁棒性实验；
- `reference_motion` 为主实验，`executed_qpos` 为辅助 ablation；
- root assist 只保留 `none` 与 `xy`；
- 各 controller 使用原生 observation、action mapping、PD 和 torque limits；
- 暂不做 matched-PD ablation；
- source-history prefill 为正式唯一初始化；
- raw policy group offset 固定 11；
- 新 controller 前 10 次推理为 self-warmup；
- warmup 轨迹和物体影响完整保存，tracking 从第 11 次统计；
- SONIC 保存真实 token/action/reference；
- 非 SONIC 的 SONIC-only 字段填 0，native telemetry 单独保存。

### 14.3 固定时序

| 组件 | 频率 | 每次 50 Hz policy 对应 |
|---|---:|---:|
| MuJoCo physics | 2000 Hz | 40 steps |
| body/hand PD | 200 Hz | 4 updates |
| policy | 50 Hz | 1 inference |
| `data.csv` | 400 Hz | 8 rows |

400 Hz 在这里是记录频率，不是 policy 或 PD 频率。所有调度使用整数 MuJoCo step，不依赖 DDS 最新消息、ZMQ 到达时间或墙上时间。

### 14.4 为什么推理时 MuJoCo simulated time 暂停

统一程序在一个 policy 边界运行同步推理，推理计算期间不推进 MuJoCo simulated time；推理完成后，在接下来的固定 40 个 physics step 中执行该 action。

这样做的优点：

- CPU 变慢只延长真实运行时间；
- reference、ghost、policy 和 physics 仍在同一仿真时间轴；
- 不会因为机器负载不同而让 action 在不同数量的 stale physics steps 后到达。

如果让 physics 在推理期间继续跑，必须额外定义异步 action delay、队列、deadline miss 和 stale observation 规则，会重新引入机器依赖。

### 14.5 ghost “提前”问题

协议 2 的一些 regular 结果中，用户看到实心机器人比 source ghost 提前约几十毫秒。当前分析是：

- 当前同步程序在仿真边界立即应用新 action；
- 原采集链是异步的，通信、调度、logger 与控制生效之间都可能存在相位差；
- 对原 CSV 的匹配直接观察到的是约 10 ms 量级的 **policy measured-state 与 CSV metadata/logger 行相位差**，它不等同于已经测得 actuator action delay；
- 所以新同步轨迹相对 source 执行轨迹可能略提前。

**[推测/待解决]** 原采集链的有效 action latency 是解释提前现象的候选原因，但尚未被直接测量。可以把确定性 action-delay queue 作为辨识实验，而不是预设修复；例如探索：

- 0 ms：当前理想同步；
- 10 ms：基于当前相位差线索的候选值，不代表已知真值；
- 20 ms：敏感性。

不建议为了模拟延迟直接恢复不可控的墙上时间异步。

### 14.6 新程序审计与已修正问题

初版统一程序出现了比旧 `change_ckpt` 更大的 root drift。逐项检查后修正：

1. **错误的额外 25 Nm SONIC body torque cap**
   SONIC release C++ 没有统一 25 Nm controller-side cap；正式实现改为使用 recording scene actuator `ctrlrange`，这批场景的踝关节可到 50 Nm。

2. **Teleopit torque limit 保持原生**
   Teleopit 使用其 tracker 原生每关节上限，并与 scene 上限取更严格者；显式传 25 Nm 仍可做诊断，证明 Teleopit 路径没有被 SONIC 修复破坏。

3. **angular velocity 定义分离**
   SONIC 使用 release MuJoCo bridge 的 free-joint `qvel[3:6]`；Teleopit 使用 pelvis-link local angular velocity。

4. **手部力矩**
   正式默认 `sonic_release=[2.45, 0.7×6] Nm`；约 1.4 Nm 的 staged XML 仅作 sensitivity。

5. **timeline quaternion**
   free/ball joint quaternion 使用 shortest-arc SLERP 并归一化，不再普通线性插值。

6. **fall 与终点**
   `qpos[2] < 0.2 m` 后锁存 fall，但固定任务时长仍继续；tracking 只统计 fall 前，任务成功资格置 false。root assist 使用半开区间 `[0,T)`，隐藏终点不再额外 teleport。

7. **viewer 隔离**
   viewer 使用独立 model/data clone，只接收正式 physics state，不能反向修改轨迹。

8. **provenance 与输出认证**
   加入模型、scene、源码、输入、输出 hash，事务式写目录和 `run_complete.json`。

9. **固定输入 parity**
   保存 encoder/decoder 精确输入，可离线重跑 ORT 并比较输出。

修正后统一测试曾达到 107 项通过；后续官方 regular 接口清理也通过相应 launcher 与统一测试。

### 14.7 ORT 与 TensorRT

当前统一程序使用：

```text
ONNX Runtime CPUExecutionProvider
```

它没有在使用 TensorRT。选择 ORT 的工程原因是：

- 同一 Python 进程可统一 regular、low-latency、v1.1 和 Teleopit；
- 当前机器可用 CPU provider，容易做固定输入调试；
- 避免 DDS/ZMQ 增加时序变量。

但“TensorRT 必须配 DDS/ZMQ”是错误理解。Python 完全可以直接加载 TensorRT engine 并在单进程固定步程序中推理。后续若 GPU 条件允许，可增加 in-process TensorRT backend，并用已保存的同一输入做 ORT/TRT parity。

当前 `--backend tensorrt` 尚不可用，程序会明确失败，而不会假装降级。

### 14.8 输出格式

每个成功结果包含：

| 文件 | 用途 |
|---|---|
| `data.csv` | 与 source 相同 header/列顺序的 400 Hz 新物理轨迹 |
| `policy_telemetry.npz` | 50 Hz controller-native observation/history/action/target/torque |
| `source_timeline.npz` | 每个输出样本的 phase-matched source state 与 provenance |
| `contact_telemetry.npz` | 400 Hz contact 数量与法向力摘要 |
| `prepared_reference.npz` | controller-neutral reference、joint order 和来源 |
| `metrics.json` | tracking、root、fall、torque、接触和任务物体摘要 |
| `run_manifest.json` | controller/model SHA、代码、scene、频率与 CLI |
| `data_schema.json` | CSV 字段分组和真实语义 |
| `source_history_prefill.json` | 实际使用的 source history |
| `source_history_context.json` | raw offset、phase match、action 转换 provenance |
| `model_snapshot/` | replay 场景和资产链接 |
| `run_complete.json` | 最后写入的完成证书与主要文件 hash |

SONIC 结果：

- 真实保存 token 64；
- 真实保存 last/raw action；
- `reference_motion[0:640]` 保存 encoder 实际使用的 reference；
- 同一次 50 Hz inference 在 8 个 400 Hz 行中重复。

Teleopit 结果：

- SONIC-only token/last/raw/reference 字段清零；
- 真实 167 observation、10×167 history 和 native action 存在 telemetry；
- `policy_received_dof_pos[0:43]` 仍保存 controller-neutral 的真实 measured body+hands state。

因此新 `data.csv` 能被现有 replay 读取，但用于训练或分析时必须同时读取 `data_schema.json` 和 telemetry，尤其不能把 Teleopit 的零 token 当作有效 SONIC token。

### 14.9 统一方案之外的 `sim_reference_overlay`

在 checkpoint/controller replacement 之后，又增加了 `sim_reference_overlay/`，用于正常 sim2sim 时同时显示 controller 正在消费的 reference：

- 实心机器人仍是正常闭环 MuJoCo 状态；
- 半透明绿色 ghost 订阅 deploy 的 `g1_debug` ZMQ 数据；
- ghost 只影响显示，不参与碰撞、PD 或 robot state；
- 默认 root mode 为 `reference`，使用首帧对齐后的 reference root；
- `actual` 模式可把 ghost translation 锁到实际机器人，便于只看姿态差异。

当前准备的 deploy CSV 数据包括：

- `csv_dance/dance`：6574 个 50 Hz frame；
- `csv_dun/dun`：1551 个 50 Hz frame；
- 均为 29 个 joint 和 14 个 selected body 的六类 CSV + metadata/info。

另有 `dun_real_safe_stage1`、near-full 及 CSV→NPZ 工具。这些 staged motion 是渐进式测试资产，**不是安全认证**，更不能因为仿真可运行就直接视为真实机器人安全。

---

## 15. 七份标准 recording 与已有统一实验

标准批次：

```text
20260720_144342_g1_sim
20260612_144127_g1_sim
20260612_144154_g1_sim
20260612_144214_g1_sim
20260722_154958_g1_sim
20260722_145020_g1_sim
20260722_160121_g1_sim
```

统一程序初版曾对七份数据运行 regular + `reference_motion` + no root assist。记录的初版 tracking 摘要如下；这些数值用于回顾，不应与后续 protocol2 修正结果直接混表：

| Recording | policy count | CSV rows | body RMSE | root XYZ RMSE |
|---|---:|---:|---:|---:|
| `20260720_144342` | 564 | 4512 | 0.14214 | 0.03967 |
| `20260612_144127` | — | — | 0.14996 | 0.03780 |
| `20260612_144154` | — | — | 0.14462 | 0.03191 |
| `20260612_144214` | — | — | 0.14151 | 0.04049 |
| `20260722_154958` | — | — | 0.12564 | 0.00972 |
| `20260722_145020` | — | — | 0.12796 | 0.01911 |
| `20260722_160121` | — | — | 0.12601 | 0.00664 |

审计后，`20260612_144154`、`20260612_144214`、`20260722_154958`、`20260722_145020` 又按 protocol2 重跑。当前磁盘上这四个 `_protocol2` 目录具有完整认证，但它们记录的是当时 protocol2 + 旧 1751 regular wrapper，只能作为该历史条件下的正式结果。若要分析当前官方 1762 regular，仍需用当前模型重跑并生成新的 manifest，不能只因目录名含 `_protocol2` 就沿用旧结果。

任务成功不能仅从 body RMSE 判断。抽屉任务需要检查抓取、物体进入目标区域和抽屉关闭；垃圾桶任务需要检查脚是否踩中并维持稳定；其他任务必须先定义 evaluator。

---

## 16. 当前模型、文件和 Git 状态

### 16.1 模型文件

#### SONIC regular 官方 release

| 文件 | 接口/大小 | SHA-256 |
|---|---|---|
| `change_ckpt/models/regular/model_encoder.onnx` | 1762 → 64；约 50.1 MB | `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3` |
| `change_ckpt/models/regular/model_decoder.onnx` | 994 → 29；约 40.9 MB | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| `change_ckpt/models/regular/observation_config.yaml` | 官方 config | `466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b` |

当前官方 1762 维 adapter 的 reference 区段为：

```text
q          [4:294]   = 290
dq         [294:584] = 290
reserved   [584:601] = 17 zeros
orientation[601:661] = 60
```

CSV/telemetry 中仍可用 canonical 640 维语义记录 q、dq 和 orientation。

#### SONIC v1.1

| 文件 | 接口 | SHA-256 |
|---|---|---|
| encoder | 1751 → 64 | `fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae` |
| decoder | 994 → 29 | `34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509` |
| config | — | `4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0` |

#### Teleopit v0.5

- `track_g1.onnx`：current obs 167 + history `10×167` → action 29；
- SHA-256：`1ebd341d9193e1c49a986450f6043ba1a9473ad46636ce0bcb1c7755c856e0de`。

#### Planner

重复 planner 已清理，只保留实际使用的：

```text
change_ckpt/models/planner/target_vel/V2/planner_sonic.onnx
change_ckpt/models/planner/target_vel/V2/planner_planner_sonic.trt
```

当前 planner ONNX SHA-256 为 `39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea`。

旧 `change_ckpt/observation_config_sonic_release.yaml` 已删除，所有引用已切换到官方 regular config。

#### 当前推理环境

- `.venv_sim`：MuJoCo 3.10.0；
- `.venv_replay`：MuJoCo 3.2.0；
- `Teleopit_rollout/.venv`：MuJoCo 3.10.0、ONNX Runtime 1.22.1；正式 rollout 实际选择 `CPUExecutionProvider`。环境还列出 `AzureExecutionProvider`，但没有 `CUDAExecutionProvider`。

### 16.2 当前模型与历史结果不能混淆

这是当前磁盘上最重要的 provenance 陷阱：

- 当前 `change_ckpt/models/regular/` 已经是 NVIDIA 官方 **1762 维** encoder/config；
- 当前七个 `change_ckpt/data/*_regular` 和七个 `change_ckpt_track/data/*_regular`（不含另存的 `_old`）的 manifest 都仍指向已删除的旧 `observation_config_sonic_release.yaml`，并记录早期 **1751 维 wrapper** 的 encoder/decoder hash `c6bd…` / `6309…`；
- 当前磁盘上四个 `controller_replacement/data/*_protocol2` 结果也都由提交 `3a41663` 附近的旧 1751 regular 模型生成；
- `39272d0` 之后虽然代码和默认模型已经切到官方 1762 接口，旧结果目录不会因此自动变成新模型结果。

因此，任何正式表格都必须读取每个结果自己的 `run_manifest.json` / `launch_manifest.json`，按模型 SHA、config SHA、代码 commit 和协议版本分组。若要宣称“官方 regular 1762 模型的结果”，需要用当前代码和当前模型重新运行，不能仅按目录名含有 `_regular` 判断。

另外，`change_ckpt/SONIC_V1_1_REFERENCE_RESULTS.md` 所列的六个标准 v1.1 输出目录，以及 `change_ckpt_track/data/sonic_v1_1_comparison/`，当前都已不在磁盘。Markdown 结果表仍是历史记录，但已经不能重新做 artifact/hash 复核。

### 16.3 Git 分支和远端

创建本文前的仓库快照：

- 当前分支：`custom/checkpoint-rollout`；
- HEAD：`ae06523 Add staged real-robot motion tooling`；
- 与本地缓存的 `origin/custom/checkpoint-rollout` remote-tracking ref 一致；
- 本地 `main`、`origin/main`、`upstream/main` 均指向 `1983e88`；
- 当前分支相对本地记录的 `upstream/main`：ahead 16、behind 0；
- `origin`：用户 fork `QingfangZhang/GR00T-WholeBodyControl`；
- `upstream`：NVIDIA `NVlabs/GR00T-WholeBodyControl`；
- 备份 tag：`backup/pre-upstream-sync-20260730` → `fde3dda`；该 tag 保留同步前的旁支历史，`fde3dda` 不是当前 HEAD 的祖先。

注意：这里的 `origin` 和 `upstream` 都是本地 remote-tracking ref 快照；本轮没有重新 `git fetch`，所以不能把它写成“已经确认 2026-09-02 服务器上的远端实时状态”。创建本文后，本文本身目前是新的未跟踪文件，因此工作树不再是完全 clean。

`.gitignore` 与 `.git/info/exclude` 的区别也在对话中确认过：

- `.gitignore` 若被提交，其规则会随 clone 成为仓库共享的默认忽略规则；它不改变已经 tracked 的文件，也可被否定规则或 `git add -f` 覆盖；
- `.git/info/exclude` 只在当前本地仓库生效，不提交，适合个人临时忽略。

当前实验目录中的下载模型、生成的 ONNX/TRT engine、实验 `data/`、sample data、虚拟环境和大部分 build 产物都被相应规则忽略；仓库中仍可能有特定模型文件的例外规则。**代码已经 push 不代表本地下载模型或实验结果也已经 push。**

### 16.4 主要提交时间线

| Commit | 内容 |
|---|---|
| `78097f7` | 旧 checkpoint reward 兼容 |
| `ff8d674` | 修复 `run_sim_loop.py` 重复 channel 初始化 |
| `1f4682a` | 增加 `change_ckpt/` |
| `067272d` | 增加 `change_ckpt_track/` |
| `8ba131d` | 增加 `run_all.sh` |
| `89c0b74` | 增加 `Teleopit_rollout/` |
| `785b8bf` | 更新 batch rollout |
| `81b3fff` | 合并官方更新，包括 SONIC v1.1/live camera |
| `8f83d90` | 增加 v1.1 qpos-track |
| `ec45d5b` | 增加 v1.1 结果记录 |
| `508e1dd` | 增加 source-history prefill 与 Teleopit hybrid |
| `fd28b1b` | 增加统一 `controller_replacement/` |
| `3a41663` | 固化确定性协议、手控制、provenance、replay 合约和测试 |
| `39272d0` | 切换为官方 SONIC regular 1762 接口 |
| `a2542ba` | 增加 `sim_reference_overlay/` |
| `ae06523` | 增加 staged real-robot motion 工具和 CSV→NPZ |

按 `git diff upstream/main...HEAD` 的三点比较口径，当前实验分支有 113 个文件差异，约 49,201 行新增、4 行删除，绝大多数隔离在实验目录。官方已有源码中的直接修改主要是：

- `gear_sonic/eval_agent_trl.py`；
- `gear_sonic/scripts/run_sim_loop.py`；
- `.gitignore`。

模型、实验数据、虚拟环境和大部分 build 产物由 `.gitignore` 排除，不随普通 Git push 上传。

### 16.5 Codex 会话历史为什么没有直接提交

对话中曾讨论把 VS Code Codex 的历史一起推到 GitHub，后来决定暂缓。原因是原始 JSONL 是事件日志，而不是已经清洗好的 Markdown transcript。它可能包含：

- system/developer 指令；
- tool call 与大量输出；
- 本地绝对路径和环境信息；
- context compaction 产生的 summary；
- interruption/rollback 记录；
- 另存的 sub-agent 或关联 session。

只复制一个 JSONL 到另一台机器后使用 `codex resume`，可能出现内容缺失，常见原因是会话索引或关联元数据没有一起迁移、部分上下文已压缩为 summary、工具/子会话存储在别处、复制时文件仍在写入，或两台机器 Codex 版本不同。

因此更稳妥的长期保存方式是本文件这样的脱敏 Markdown：保留用户目标、代码决策、验证证据、命令和结论，不公开原始内部事件流。

---

## 17. 当前常用命令

### 17.1 统一正式 rollout

regular + 原 `reference_motion` + no assist：

```bash
Teleopit_rollout/.venv/bin/python controller_replacement/launch_rollout.py \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  --controller regular \
  --reference-mode reference_motion \
  --root-assist none
```

更换 controller：

```text
--controller low_latency
--controller sonic_v1_1
--controller teleopit
```

qpos-track ablation：

```text
--reference-mode executed_qpos
```

root oracle ablation：

```text
--root-assist xy
```

批量运行：

```bash
controller_replacement/run_experiments.sh \
  sample_data/ztj/20260612/20260612_144127_g1_sim \
  sample_data/ztj/20260612/20260720_144342_g1_sim
```

### 17.2 查看结果

下面的 native C++ replay 依赖 MuJoCo 3.2 和 GLFW shared library。普通 shell 下当前 `ldd` 仍会显示 `libmujoco.so.3.2.0`、`libglfw.so.3` 未找到，因此运行前必须正确配置动态库搜索路径；仅激活 Python venv 不一定会改变 native ELF 的查找路径。

只看新 qpos：

```bash
./sample_data/ztj/replay_mujoco_csv \
  controller_replacement/data/<result-folder>
```

同时看 phase-matched source ghost：

```bash
.venv_replay/bin/python controller_replacement/replay_mujoco_compare.py \
  controller_replacement/data/<result-folder>
```

先做无 GUI 校验：

```bash
.venv_replay/bin/python controller_replacement/replay_mujoco_compare.py \
  controller_replacement/data/<result-folder> --dry-run
```

### 17.3 `sim_reference_overlay` 的 deploy 命令

在 `gear_sonic_deploy/` 中：

```bash
export MOTION_NAME=dance

just run g1_deploy_onnx_ref lo \
  ../change_ckpt/models/regular/model_decoder.onnx \
  ../sim_reference_overlay/data/csv_${MOTION_NAME} \
  --obs-config ../change_ckpt/models/regular/observation_config.yaml \
  --encoder-file ../change_ckpt/models/regular/model_encoder.onnx \
  --planner-file ../change_ckpt/models/planner/target_vel/V2/planner_sonic.onnx \
  --input-type manager \
  --output-type zmq \
  --enable-csv-logs \
  --disable-crc-check
```

用户原命令与此核心相同；这里只显式补了 `--output-type zmq`。当前它通常也是默认值，但写出来更利于复现。

注意：

- `MOTION_NAME` 应先 `export`，不要在同一行临时赋值后立即期待 shell 展开 `${MOTION_NAME}`；
- `lo` 适用于 simulator 与 deploy 在同一 network namespace；
- `--input-type manager` 通常需在 deploy terminal 用按键开始；
- `--enable-csv-logs` 是 deploy 拆分日志，不等于完整任务 replay `data.csv`。

---

## 18. 论文、新数据集和研究价值讨论

### 18.1 仅“换 controller replay”够不够

对话形成的判断是：仅证明“另一个 controller 可以 replay 一条轨迹”，很难单独构成 ICRA/ICLR 论文，因为已有工作已经涉及动作重放、控制模式转换、sim2sim 和 controller mismatch。

更有价值的切入点是：

> 接触型 humanoid whole-body demonstration 是否能够在不重新遥操、不联合训练 controller 的情况下，迁移到独立训练的 frozen WBC；哪些因素决定迁移成功；迁移后的数据是否仍对 VLA 微调有效。

可能的贡献包括：

- controller replacement benchmark；
- controller-neutral reference/adapter；
- contact-aware 可迁移性指标；
- 失败归因：history、heading、root、PD、delay、contact；
- 自动筛选仍可用的迁移数据；
- 对下游 GR00T/VLA 微调收益的实证。

ICRA 更自然地接受系统、控制、接触和机器人评测；若投 ICLR，则需要更强的学习方法，例如 learned controller-invariant representation/adapter 和跨 controller 泛化。

### 18.2 还需要补什么实验

- 更多任务类别和 recording；
- 每个条件多次重复，报告成功率和置信区间；
- 多个 SONIC checkpoint 与至少一个非 SONIC controller；
- source×target controller matrix；
- task-specific evaluator，而不是只看终点位姿；
- contact、torque、root、tracking 与成功率的关联；
- root assist 只做 oracle 上界；
- 对同一数据进行 GR00T/VLA 微调，验证数据价值；
- 条件允许时做真实机器人验证。

### 18.3 HumanoidEveryday

这个数据集可能提供 G1 的身体、手、odom、IMU、图像或触觉状态，可用于离线构造 SONIC/Teleopit reference 或做 pseudo-label。

但核心问题是：如果缺少完整 scene mesh、物体初态、摩擦、接触参数和控制时序，就不能在仿真中可靠判断“换成 SONIC 后是否仍抓起物体”。因此：

- 仅做 kinematic relabel 的数据不能自动获得“任务成功”标签；
- 可以用于预训练、轨迹先验或候选数据；
- 要作为闭环成功 manipulation 数据，仍需数字孪生重建、真实场景再执行或其他可靠验证。

### 18.4 PICO、SMPL 与 teleop encoder

这部分仍应保守：

- 官方代码存在 SMPL motion 和 VR/teleop keypoint 等不同入口；
- 三点 VR 可以通过 IK 或学习模型估计全身，但不是全身每个关节的直接测量；
- 增加 PICO body trackers 可以提高全身姿态估计；
- 由于实际采集脚本缺失，当前数据究竟走 SMPL encoder 还是 teleop encoder 尚未最终确认。

训练配置中讨论过 `freeze_frame_aug`：它会对一定比例的 motion 在随机位置后保持到片段末尾，持续时间取决于冻结位置，而不是固定秒数。它是训练增强，不能在没有采集源码证据时被用来解释 CSV future slot 重复。

---

## 19. 当前尚未解决的问题

1. **原始采集链**：缺少实际数据采集脚本，PICO/SMPL/teleop encoder、手命令映射和 publisher lookahead 仍未完全确认。
2. **future slot 重复根因**：现象已在多份 CSV 中验证，但没有采集端源码给出最终解释。
3. **逐值复现 source regular**：CSV 没保存完整 solver/contact/warm-start/DDS phase，所以即使同一 checkpoint 也不能保证 bitwise/trajectory 完全一致。
4. **时相/有效 action latency**：统一同步程序相对原异步 source 有时视觉上提前约几十毫秒；当前只直接测到 policy received-state 与 CSV logger/metadata 的相位差，尚未测出真实 actuator action latency。确定性 delay queue 可用于辨识，但尚未加入正式协议。
5. **ORT/TensorRT parity**：当前正式统一结果来自 ORT CPU；in-process TensorRT backend 尚未实现。
6. **任务 evaluator**：多数任务仍依赖人工 replay，物体发生运动或接触不等于任务成功。
7. **root assist 的科学解释**：xy assist 是 oracle，必须与 native controller 结果分表；不能以 assisted success 证明 controller 独立能力。
8. **统计规模**：目前数据量、controller 数量和重复次数不足以形成可靠成功率结论。
9. **VLA 数据价值**：轨迹“看起来可用”不等于能提升 GR00T/VLA；仍需实际 finetune/evaluation。
10. **外部数据场景恢复**：HumanoidEveryday 等数据若没有完整任务环境，接触成功无法可靠验证。

---

## 20. 后续建议的实验顺序

1. 固定当前 `controller_replacement` protocol2，不再同时改变 reference、PD、history 和 timing。
2. 对七份标准 recording 分别运行：
   - regular / low-latency / v1.1 / Teleopit；
   - `reference_motion`；
   - root assist `none` 与 `xy`。
3. 每个条件至少重复 3–5 次；固定随机性并记录机器、backend、模型和 scene hash。
4. 先量化 source/action/接触之间的 cross-correlation，再把确定性 action delay（例如 0/10/20 ms）作为辨识性 ablation，不能预先把 10 ms 当作真实延迟。
5. 为抽屉、垃圾桶和其他场景编写 task-specific evaluator。
6. 分别报告：
   - stability/fall；
   - body/anchor/root tracking；
   - hand tracking；
   - contact/torque；
   - task success；
   - assist 依赖。
7. 再运行 `executed_qpos` ablation，解释 reference intent 与旧执行轨迹的差异。
8. GPU 条件具备时加入 TensorRT parity；不要重新引入非确定性 DDS/ZMQ 才能使用 TensorRT。
9. 从自动 evaluator 通过的迁移轨迹中构造训练集，最后验证 GR00T/VLA 微调收益。

---

## 21. 术语表

| 术语 | 本项目中的含义 |
|---|---|
| source recording | 原始采集成功轨迹及其场景、reference、手目标和 qpos |
| controller replacement | 保持任务上下文，换另一个 body whole-body controller 闭环执行 |
| `reference_motion` | SONIC 采集时的上游动作意图；正式主 reference |
| `executed_qpos` | 旧 controller 实际执行轨迹；qpos-track reference |
| token | SONIC encoder 输出的 64 维离散/量化 latent |
| raw action | decoder/tracker 的 29 维原始输出，尚不是实际 qpos |
| q target | raw action 经 scale/default/clip 后的关节位置目标 |
| qpos | MuJoCo 实际广义位置，包括机器人和场景物体 |
| source-history prefill | 用 takeover 前的真实 source history 初始化新 controller |
| self-warmup | 新 controller 接管后，用自身 action 逐步替换旧 history 的阶段 |
| recorded lag | 从 CSV slot 数值匹配推断出的实际 future policy offset |
| canonical lag | observation config 名义定义的 future offset |
| root assist xy | 运行时硬对齐 root x/y 与 vx/vy 的 oracle |
| ghost source | 原 recording 同一时刻的实际 qpos，可视化用 |
| ghost reference | 构造给 tracker 的 reference pose，可视化用 |
| ORT | ONNX Runtime；当前统一程序实际使用的 backend |
| TensorRT | NVIDIA 推理 backend；旧 C++ deploy 使用，统一程序尚未接入 |
| task success eligible | 轨迹完整、未跌倒且满足继续进行语义成功判定的资格，不等于已成功 |

---

## 22. 本纪要的证据边界

本文件将“当前代码行为”和“历史实验结论”尽量分开：

- 代码、模型和 Git 状态以 2026-09-02 本地工作区为准；
- 旧实验可能由早期协议、旧模型或旧 torque cap 生成，必须读取各自 manifest，不能只按目录名判断；
- 用户在 viewer 中确认的任务现象被保留为用户观察，没有自动 evaluator 时不提升为统计结论；
- 外部项目和论文的内容只用于形成研究方向，最终论文写作仍需重新查阅并引用原始来源；
- 这份 Markdown 是从会话和仓库资料整理出的可读记录，不应把原始 Codex JSONL 直接推送到公开仓库。原 JSONL 还可能包含系统指令、工具输出、环境路径和被压缩的上下文，不是稳定的可移植聊天导出格式。
