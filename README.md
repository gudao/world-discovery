# World Discovery · MuJoCo

一个研究「智能体如何在未知环境中主动试探、区分竞争解释、并按需扩展自己的世界模型」的实验项目。载体是 MuJoCo 桌面方块（隐藏质量/摩擦/固定状态），方法路线是显式贝叶斯系统辨识 + 信息增益试探 + 预注册判据的严格实验协议。

**当前状态（2026-10-10）：** v0.1 基线已冻结交付；v0.2 协议已冻结，阶段 1（区分测试）完成——机制链按设计工作，但环境区分过便宜，下一步是阶段 4 环境升级。详见下方研究进展日志。

## 项目愿景

长期目标是回答一个问题：**一个系统如何从"已有的世界观解释不了眼前现象"出发，自主地设计试探、区分竞争解释、修正自己，并且在必要时提出新的世界变量——而不只是存更多数据？**

我们把它拆成一条可检验的设计原则链：

1. **识别未知**：系统持续察觉"当前表征无法稳定解释后果"（持续失配可被检出）；
2. **主动试探**：在解释集合 H 上计算信息增益，挑选能最大程度区分竞争解释的动作，而不是盲目探索或固定扫描；
3. **快速修正**：一次有信息量的交互足以大幅收缩假设（v0.1 已确认：1–2 次交互把预测误差压到 ~0.04）；
4. **按需扩展**：所有已知解释持续失败时，用 MDL 式判据在接受新变量（u = g_φ(h)，可从少量历史推断、可迁移）与更廉价的解释（更新参数 / 延长记忆 / 改噪声模型）之间做出有原则的选择。

这条链的每一环都要用预注册判据单独检验，负结果如实记录——**本项目的产出是经过严格检验的机制结论，不是刷指标的演示**。

迭代路线（详见 `docs/ROADMAP.md`）：v0.1 机制基线（已完成）→ v0.2 显式假设集合与区分/迁移/新区分三判据（进行中）→ v0.3 可学习表征 → v0.4 多模态主动感知（视觉/触觉/传感成本）→ v0.5 表征扩展与知识整合。终极检验标准是区分测试、迁移测试（排除"多存数据"解释）与新区分测试（系统能否自主发现并接受正确的世界新变量），而非单纯的预测误差。

## 试验目的

每个版本回答一组预先冻结的研究问题。当前 v0.2 的三个问题（完整定义见 `docs/PROTOCOL-v002.md`）：

- **R1'（区分）**：显式假设后验上的信息增益试探，能否把"旧表征视为相同、真值不同"的世界对区分开，并优于随机与固定扫描？
- **R2'（诊断与修正）**：持续失配被检出后，系统能否在并列候选（更新参数 / 延长记忆 / 改噪声模型 / 新增变量）中按预注册判据接受正确的一个？
- **R3'（新区分的迁移）**：被接受的新变量能否从少量历史推断、并迁移到同分布新世界（排除"多存数据"解释）？

v0.1 的 A/B 实验（试探选择是否有用、按需容量扩展是否有用）作为机制基线保留，其全部输出冻结只读；v0.1 的负结果（主动评分的动作坍缩）归因于假设集合 H 未被显式表示，已在 v0.2 中替换为显式机制后正面检验。

> **给协作者：实验日志规范。** 每次实验（含负结果和未完成阶段）结束后，必须更新下方「研究进展日志」——写清楚当时的问题、思路、完整步骤（含命令与输出目录）、结论（含置信区间/预注册判定）和下一步。原则：先冻结协议再运行，绝对数字 + 配对 CI 优先于均值排名，负结果如实记录。协议细节见 `docs/PROTOCOL.md`（v0.1）与 `docs/PROTOCOL-v002.md`（v0.2，已冻结）。

## 研究进展日志

### 2026-10-10 · v0.2 阶段 1：区分测试（当前最新）

**问题。** v0.1 只验证了"机器"（顺序/门控/成本/协议）而未验证实质：假设集合 H 从未被显式表示（ensemble 输出方差是采样噪声，不是竞争解释）。阶段 1 检验：显式假设集合上的信息增益试探，能否把"旧表征视为相同、真值不同"的世界对区分开（协议 R1'，判据见 `docs/PROTOCOL-v002.md` 第 3 节）。

**思路。** 用解析 Coulomb 假设族 `(m, μ_k, μ_s, fixed)` + 200 粒子后验显式表示 H；试探评分 = 矩匹配信息增益 − 推力成本（λ 预注册校准为 10:1）；ε=0.15 预算用于验证性复测。对照策略 random / fixed / v0.1-active（后者原样冻结作为"代理失败"阴性对照）。

**步骤。** 协议 v0.2.0 冻结 → σ 由训练残差校准（冻结 σ = [0.0040, 0.0424]）→ 5 种子 × 40 世界对（friction/mass/fixed 三型），每世界 12 次真实试探。复现命令与全部数值：`runs/v002-discrimination/`（records.json、score_log.jsonl、summary.json），分析报告 `runs/v002-discrimination/report.md`。

**结论。** **环境过便宜条款触发**（fixed 0.995 ≥ 0.95），不做策略排名。机制链按设计工作：IG 平均 2.1 次试探饱和（首步 ±12，IG≈6 nats），推力成本为 fixed 的 53%、v0.1-active 的 30%；v0.1-active 的 ±12 动作坍缩病理在 v0.2 读者下复现（确认为评分目标错误，非环境难度）。异常线索：seed11/pair8 fixed 序列下粒子自信坍缩于错误参数（顺序试探的族误差模式）。**下一步 = 协议第 7 节阶段 4：环境升级让区分成为瓶颈（更小 δ + 观测噪声 + 推力成本 + 静/动摩擦分离），环境参数先进协议修正案冻结再跑。**

### 2026-10-10 · v0.1 基线：实验 A（试探选择）与实验 B（按需扩展）

**问题。** 拆开验证两件事：主动试探策略是否优于随机/固定；容量扩展判据是否有效。

**步骤。** `world-discovery run --config configs/quick.json`（seed 42，12 测试世界），A/B 两实验，输出在 `examples/quick-A`、`examples/quick-B`，汇总 `examples/RESULTS.md`。

**结论（单种子，配对差 + 95% CI，我方复核）。** 局部残差记忆是全部收益来源（active/none 不随预算变化，1–2 次交互压 NMSE 至 ~0.04）；「主动>随机」在 CI 下不成立（A：−0.064 CI[−0.157,+0.029]；B：−0.030 CI[−0.080,+0.020]，均 n.s.）；B 中 fixed 显著优于 active（−0.0144 CI[−0.026,−0.003]）且推力成本仅一半；机制诊断：active 动作分布坍缩到 ±12（argmax 分歧评分在外推区最高导致覆盖崩塌），cost_weight=1e-4 被方差项淹没。wide 在摩擦漂移下有效但 grow 判据几乎不触发——扩展判据失效 ≠ 容量无收益，两者要分开。**v0.1 全部输出冻结，只读。**

## 快速开始

Python 3.10+，CPU 即可，无需 GPU。推荐使用独立虚拟环境。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 改用：.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q

world-discovery run --config configs/quick.json --out runs/my-A --experiment A
world-discovery run --config configs/quick.json --out runs/my-B --experiment B
```

打开输出目录的 `report.html` 查看曲线和表格。没有图形桌面的服务器也能完成全部实验；实验不调用渲染器。

在部分 BLAS 默认开很多线程的电脑上，小网络可能反而更慢。Linux / macOS 可以先执行：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Windows PowerShell 对应 `$env:OPENBLAS_NUM_THREADS="1"`、`$env:OMP_NUM_THREADS="1"`。

桌面查看物理场景：

```bash
world-discovery view
# macOS 的 MuJoCo 被动查看器通常需要：
mjpython -m world_discovery.cli view
```

查看器是固定推力演示，不是智能体回放。它需要本机图形环境；本次交付验证的是无渲染实验。

## 项目结构

```text
configs/                 quick/full 实验配置
src/world_discovery/
  env.py                 MuJoCo 接触环境、观察接口、隐藏世界参数
  model.py               NumPy 神经网络集成、局部记忆、容量扩展判据
  explorer.py            随机、固定、主动三种动作策略
  experiment.py          数据生成、训练、配对评估、审计
  report.py              曲线与离线 HTML 报告
  cli.py                 命令入口
  viewer.py              可选桌面场景演示
tests/                   物理行为、状态恢复、学习和数据隔离测试
docs/PROTOCOL.md         数学定义与实验约束
docs/ROADMAP.md          迭代路线和未解决问题
examples/                本次已实际运行的 A/B 结果
```

## 环境和数据

场景是桌面上的方块。质量、摩擦和是否固定是隐藏条件；MuJoCo 处理真实接触。为控制实验复杂度，方块有 x/z 两个滑动自由度，禁止旋转和侧向运动。

- 模型观察：理想位置、理想速度。速度直接来自仿真传感状态，**不是视频估计**。
- 模型输入：当前速度、推力、动作持续时间；位置具有平移不变性，不进入预测网络。
- 模型输出：下一时刻的位置增量、速度增量。
- 动作：等待或沿 x 方向施加不同大小、符号的推力，持续时间相同。
- 隐藏真值：只保存到 `evaluator_only_worlds.json`，学习器和探索器不接收环境对象或真值。

不需要下载数据集。`collect` 自动生成连续交互轨迹，包含 observation/action/next_observation/episode_id。环境身份分割训练、校准和测试；连续参数随机采样，不意味着所有潜在物理因子都完全不重叠。

## 实验 A：试探的选择有没有用？

预训练和测试均包含不同质量、摩擦和固定状态。比较：

| 策略 / 记忆 | 用途 |
| --- | --- |
| random / small | 随机动作 + 小型局部适应 |
| fixed / small | 固定动作序列 + 相同局部适应 |
| active / small | 不确定性评分动作 + 相同局部适应 |
| active / none | 去掉局部修正，隔离快速记忆的作用 |

主要问题：相同真实交互次数，哪种策略对统一测试问题预测得更好？主动评分是集成分歧 + 岭回归杠杆值 - 推力成本，保留一定概率随机探索。**不是精确的贝叶斯信息增益。**

所有动作持续时间相同，交互次数对应相同模拟时间；不同策略的推力消耗仍可能不同，因此单独记录 `force_time_effort`。

## 实验 B：按需扩展有没有用？

预训练和校准只接触摩擦系数 0.25；测试引入不同摩擦。增加：

| 记忆 | 机制 |
| --- | --- |
| small | 始终使用小型残差适配器 |
| wide | 一开始就启用更多特征 |
| grow | 持续失配后，用历史中的保留段比较候选，决定是否启用更多特征 |

随机和主动策略都比较 small/wide/grow。所有容量候选共享同一特征库、正则项和数据，避免差异来自完全不同的初始化。

**本版扩展的是局部残差预测器的特征容量。** 全局 tanh MLP 通过 Adam 训练；局部特征是固定随机投影，读出权重通过岭回归更新。启用更多随机特征不等于自动发现摩擦概念，也不等于学习到了新的物理本体。这个实现用于先检查扩展判据、对照协议和计算代价。

## 管理迭代

每次修改复制一份配置，例如 `configs/v002.json`，使用新的输出目录。默认拒绝覆盖已有实验，确实需要重跑时加 `--overwrite`。配置保存在结果目录，后续阶段要求与原配置一致。

```bash
# 分阶段调试
world-discovery collect --config configs/quick.json --out runs/staged --experiment A
world-discovery train --config runs/staged/config.json --out runs/staged --experiment A
world-discovery evaluate --config runs/staged/config.json --out runs/staged --experiment A

# 正式比较至少运行多个随机种子；每个种子单独留档
world-discovery run --config configs/full.json --out runs/full-A --experiment A --seeds 42 43 44
world-discovery run --config configs/full.json --out runs/full-B --experiment B --seeds 42 43 44
```

多种子命令逐个执行，便于在普通电脑运行。每个种子有独立报告；当前未实现跨种子自动合并统计。不要把一个种子内的世界标准误当成跨训练种子的置信区间。

建议每次协议变更提交一次 Git，并在每次实验结束后更新顶部「研究进展日志」，再提交对应的 `runs/` 输出目录。

## 每次实验留下什么？

| 文件 | 内容 |
| --- | --- |
| config.json / protocol.json | 完整配置、实验定义 |
| train.npz / validation.npz | 不含隐藏参数的训练与校准轨迹 |
| evaluator_only_worlds.json | 评估侧隐藏条件，禁止训练读取 |
| model.npz | 全局网络权重、输入输出归一化统计 |
| calibration.json | 校准阈值、训练与校准损失 |
| interactions.jsonl | 每次真实动作、观察、预测误差 |
| metrics.csv | 每世界、每策略、每预算的指标与成本 |
| growth_events.json | 每次扩展测试、增益和接受/拒绝结果 |
| summary.json | 学习曲线与最终配对差值 |
| learning_curve.png / report.html | 离线结果展示 |
| manifest.json | Python/依赖版本、数据/模型/源代码 SHA-256 |

测试查询由评估器从统一初始条件构造，在各策略之间完全一致。评估器可以恢复 MuJoCo 状态获得反事实标签，**探索器不能调用这个接口或读取测试标签**。测试标签不用于更新局部记忆、选择扩展或调整阈值。

最终配对差为 `random NMSE - active NMSE`，正值表示本次主动策略更好。NMSE 用训练输出标准差归一化，不是传统 R²。原始位置/速度 MSE 也同时保存。

## 当前边界

- 不包含视频视觉、多模态传感、物体分割、机械臂控制。
- 不具备跨世界共享新概念的机制；每个世界的局部适配器都会重置。
- 不确定性和扩展阈值都是待校准的工程近似。
- 自动扩展选择使用在线历史保留段；重复使用这段数据会产生选择偏差，独立最终评估可揭示但不能消除它。
- 扩展之后参数可能比 small 多；必须比较 wide，并报告成本。
- 该小场景只能验证机制的一部分，不能据此宣称通用智能。

官方参考：

- MuJoCo Python：https://mujoco.readthedocs.io/en/latest/python.html
- Dreamer：https://arxiv.org/abs/1912.01603
- Plan2Explore：https://proceedings.mlr.press/v119/sekar20a.html
- MDL：https://arxiv.org/abs/math/0406077
