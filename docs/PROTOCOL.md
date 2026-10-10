# 实验协议 v0.1

## 研究问题

R1：相同交互预算下，选择能区分预测的动作是否优于随机与固定动作？

R2：发生持续失配时，按需扩大局部预测器是否比固定小容量有效，且相对于始终大容量有成本收益？

R3（未实现）：能否增加可迁移的、历史可推断的隐变量，而不是只增加拟合容量？

## 当前数学实现

输入 x=(v,F,Δt)，标签 y=(Δposition,Δvelocity)。所有归一化均只用训练数据统计。

全局模型是 K 个 bootstrap 训练的 tanh MLP：

`f_k(x)=tanh(normalize(x) W1_k+b1_k) W2_k+b2_k`。

输出位于归一化坐标。局部特征为：

`phi_d(x)=[1, normalize(x), tanh(normalize(x) R[:,:d]+b[:d])]`。

对每个成员拟合残差读出：

`B_k=(Phi.T Phi+lambda I)^(-1) Phi.T (Y-f_k(X))`。

实现使用 `numpy.linalg.solve`，不显式求逆。局部记忆至多 probe_budget 个样本，世界结束即丢弃。

预测为集成均值 `mean_k[f_k(x)+phi_d(x) B_k]`。

主动动作评分：

`score(a)=mean_output Var_k prediction_k(x_a)+0.01 leverage(x_a)-cost_weight |F| Δt`。

`leverage=phi (Phi.T Phi+lambda I)^(-1) phi.T`。

噪声为零的初始基线仍有模型失配。未来添加观察噪声时，必须重新区分不可约随机性与知识不确定性，不能直接把总预测误差当探索收益。

扩展触发：连续至少两次 prequential NMSE 超过全局校准阈值，且交互样本达到 growth_min_samples。

历史前段拟合 small 和 wide，后 growth_holdout 个样本保留比较。接受条件：

`L_small-L_wide > growth_min_gain + growth_penalty * extra_adapter_parameters`。

两个候选都重新拟合相同训练前缀，避免旧模型未训练充分产生不公平比较。扩展后再用全部已获得交互拟合。未来应使用新到来的专门诊断数据，减少重复保留段选择偏差。

## 数据协议

- Train/validation/test 使用不同随机流和环境实例。
- A：三组来自相同参数范围，检验新实例适应。
- B：train/validation 摩擦固定，test 改变摩擦，检验摩擦分布变化。
- 测试环境中的固定世界对整体平均可能有明显影响，metrics.csv 保留 fixed 标签供分组审计，模型不可读取该字段。
- 第一版未保证各组 fixed 比例严格相等；正式实验可改成分层采样。
- 测试查询来自 3 种统一前置状态 × 候选动作，所有策略共享。
- 查询不计入智能体交互预算，也不能用于训练；它们属于模拟器评估特权，真实机器人需要独立评估流程。
- 不允许从最佳测试种子/配置挑选结果后再把它当独立验证；调参结束需要新测试世界与种子。

## 判定标准

第一轮没有预设“主动必须胜出”。如果主动不胜出，应检查评分、覆盖度、分歧校准和成本，再在新测试数据验证修正。

容量扩展的成功需要预测和成本上的可复现收益。概念发现还必须通过跨动作、跨物体、跨环境的因子迁移，以及与固定等容量模型的对照。本版不能回答概念发现问题。
