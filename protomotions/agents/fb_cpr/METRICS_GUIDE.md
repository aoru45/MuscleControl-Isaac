# FB-CPR 训练指标完整指南

## 快速概览

FB-CPR 训练包含 **14 个关键指标**，分为 4 个类别。我们的实现**完全记录了所有指标**。

### 最重要的 7 个指标

| # | 指标 | 应该怎样 | 警告信号 |
|---|------|--------|--------|
| 1 | `fb/loss` | ↓ 下降 | 平坦或上升 |
| 2 | `discriminator/loss` | ↓ 下降 | 很高或不动 |
| 3 | `losses/critic_loss` | ↓ 下降 | 上升 |
| 4 | `losses/actor_loss` | ↓ 下降/变负 | 持续很大正数 |
| 5 | `critic/q_value` | ↑ 上升 | 下降 |
| 6 | `actor/fb_q` | ↑ 上升 | 下降 |
| 7 | `fb/b_norm` | ≈ 8.0 | 偏离很大 |

---

## 1. Forward-Backward Loss (5 个指标)

这是 FB-CPR 的**核心部分**。

### `fb/loss` - FB 总损失
- **含义**: 前向映射的总损失
- **公式**: `fb_offdiag + fb_diag + ortho_coef * orth_loss`
- **期望**: 逐步下降到很小的值
- **影响因素**: `embedding_layers`, `ortho_coef`

### `fb/diag` - 对角线匹配损失
- **含义**: 同一转移序列中，当前状态和目标状态的一致性
- **公式**: `-diagonal(M - target_M).mean()`
- **初期**: 很负（因为网络还没学）
- **期望**: 逐渐上升，最终接近 0
- **意义**: 如果不上升，说明网络没学到状态转移的一致性

### `fb/offdiag` - 非对角线最小损失
- **含义**: 不同转移序列应该产生不同的潜在状态
- **公式**: `0.5 * ((M - target_M) * off_diag)^2 / off_diag.sum()`
- **初期**: 很高
- **期望**: 逐渐下降到接近 0
- **意义**: 防止所有状态被编码为同一点（平凡解）

### `fb/orth_loss` - 正交性约束损失
- **含义**: Backward map 的输出应该形成正交基
- **公式**: `-Cov.diag() + 0.5 * (Cov * off_diag)^2`，其中 `Cov = B @ B.T`
- **期望**: 从中等值逐渐下降到接近 0
- **数学意义**:
  - 对角线项 → 1（单位方差）
  - 非对角线项 → 0（不相关）

### `fb/b_norm` - Backward 映射向量范数
- **含义**: Backward map 输出的 L2 范数均值
- **期望**: 稳定在 `√z_dim`
  - z_dim=64 时，应该在 8.0 左右
  - 范围应该在 [6.0, 10.0] 之间
- **警告**: 如果严重偏离（>20 或 <3），说明归一化失败

---

## 2. Discriminator Loss (4 个指标)

判别器学习区分专家和训练数据。

### `discriminator/loss` - 判别器总损失
- **公式**: `expert_loss + train_loss + gp_coef * wgan_gp`
- **期望**: 逐步下降
- **范围**: 通常在 0.5-2.0

### `discriminator/expert_loss` - 专家识别损失
- **含义**: 判别器对专家数据的损失
- **公式**: `-log(sigmoid(logits)).mean()`
- **期望**: 下降（判别器学会识别专家）
- **含义**: 当此项小时，判别器能正确识别专家

### `discriminator/train_loss` - 训练识别损失
- **含义**: 判别器对训练数据的损失
- **公式**: `softplus(logits).mean()`
- **期望**: 下降（判别器学会拒绝训练数据）
- **含义**: 当此项小时，判别器能正确拒绝非专家

### `discriminator/wgan_gp` - WGAN 梯度惩罚
- **含义**: Wasserstein GAN 的梯度惩罚项
- **期望**: 稳定在 1 左右，不要太大
- **范围**: 应该 < 100，理想值 0.5-2.0
- **警告**: 如果 > 100，说明梯度被过度惩罚

---

## 3. Critic Loss (2 个指标)

评论家学习价值函数。

### `losses/critic_loss` - 评论家 TD 损失
- **含义**: 评论家网络的时间差分 (Temporal Difference) 损失
- **公式**: `MSE(Q_values, disc_reward + discount * next_Q_target)`
- **期望**: 逐步下降到很小的值
- **含义**: 评论家学会准确估计价值

### `critic/q_value` - 平均 Q 值
- **含义**: 评论家估计的平均累积折扣奖励
- **期望**: ↑ 缓慢上升
- **含义**: 策略学到的价值在增加
- **警告**: 如果下降，说明策略质量变差

---

## 4. Actor Loss (3 个指标)

策略网络的优化。

### `losses/actor_loss` - 策略损失
- **含义**: 演员网络的优化目标
- **公式**: `-Q_critic - reg_coeff * Q_fb`
- **期望**: 下降，最终变为负值（表示奖励）
- **含义**:
  - 第一项：最大化评论家奖励
  - 第二项：最大化 FB 表示的奖励

### `actor/fb_q` - FB 价值
- **含义**: 通过前向映射估计的价值
- **公式**: `(Fs * z).sum(-1)`
- **期望**: ↑ 缓慢上升
- **含义**: 策略在改进 FB 表示的价值

### `actor/critic_q` - 评论家价值
- **含义**: 评论家估计的价值
- **期望**: ↑ 缓慢上升
- **含义**: 策略在改进传统的价值估计

---

## 训练动态参考

### 初期 (Epoch 1-10)
- FB 损失: 高 → 中
- Discriminator 损失: 中等
- Critic 损失: 中等
- Q 值: 低
- 特点: 可能有噪声，不要太担心

### 中期 (Epoch 10-100)
- FB 损失: 中 → 低
- Discriminator 损失: 低
- Critic 损失: 低
- Q 值: 缓慢上升
- 特点: 应该看到稳定的下降趋势

### 后期 (Epoch 100+)
- FB 损失: 接近 0
- Discriminator 损失: 稳定
- Critic 损失: 很小
- Q 值: 持续上升或稳定在高值
- 特点: 所有指标应该稳定

---

## 问题诊断和解决方案

### ❌ 问题: `fb/offdiag` 完全不下降

**根本原因**: 网络容量不足，无法学到不同的表示

**解决方案** (按优先级):
1. 增加 `embedding_layers`: 2 → 3 或 4
2. 增加 `hidden_dim`: 512 → 1024
3. 减小 `lr_f` 和 `lr_b`，让训练更稳定

**诊断**: 看看 `fb/diag` 是否也在上升

---

### ❌ 问题: `discriminator/loss` 很高，不动

**根本原因**:
- 梯度惩罚太强，压制了学习
- 数据分布差异太大
- 判别器和其他网络学习速度不匹配

**解决方案**:
1. 减小 `grad_penalty_discriminator`: 10 → 5 或 2
2. 增加 `lr_discriminator`: 1e-4 → 5e-4
3. 检查专家和训练数据是否差异太大

**诊断**: 看 `discriminator/expert_loss` 和 `discriminator/train_loss` 的分别

---

### ❌ 问题: `critic/q_value` 在下降

**根本原因**: 策略质量变差，或判别器奖励信号不稳定

**解决方案**:
1. 减小 `lr_actor`: 1e-4 → 5e-5
2. 检查 `discriminator` 是否学习稳定
3. 增加 `batch_size` 以减少噪声

**诊断**: 同时检查 `discriminator/loss` 是否稳定

---

### ❌ 问题: `actor_loss` 持续为大正数（如 > 100）

**根本原因**: 策略优化失败，可能是学习率太高

**解决方案**:
1. 减小 `lr_actor` 和 `lr_f`: 1e-4 → 5e-5 或 1e-5
2. 增加 `batch_size`
3. 检查初始化

**诊断**: 看 `actor/fb_q` 和 `actor/critic_q` 是否在上升

---

### ❌ 问题: `fb/b_norm` 远离 √z_dim

**根本原因**: 最严重的问题之一，说明归一化失败

**解决方案**:
1. 检查是否正确实现了 `Norm()` 层
2. 增加 `ortho_coef`: 1.0 → 2.0 或 5.0
3. 减小 `lr_b`

**诊断**: 这通常表示初始化或网络架构问题

---

### ⚠️ 问题: 训练不稳定，损失波动大

**根本原因**: 学习率太高，或 batch_size 太小

**解决方案**:
1. 减小所有学习率: 1e-4 → 5e-5
2. 增加 `batch_size`: 256 → 512 或 1024
3. 增加 `clip_grad_norm`: 0 → 1.0

**诊断**: 看梯度范数是否过大

---

## 最佳监测实践

### 推荐监测周期

- **每步记录**: 所有 14 个指标
- **每 10 步检查**: 指标是否有明显的负趋势
- **每 100 步分析**: 整体进展，对比超参效果

### 关键指标组合

#### 组合 1: 收敛性检查
```
是否收敛? = (fb/loss ↓) AND (discriminator/loss ↓) AND (critic/loss ↓)
```

#### 组合 2: 质量检查
```
质量提升? = (critic/q_value ↑) AND (actor/fb_q ↑)
```

#### 组合 3: 稳定性检查
```
稳定吗? = 没有NaN/Inf AND 梯度范数 < 1000 AND 没有突跳
```

### 可视化建议

使用 TensorBoard 或 Weights & Biases 绘制:

1. **第一行**: 三个 loss 曲线
   - `fb/loss`, `discriminator/loss`, `losses/critic_loss`

2. **第二行**: Q 值曲线
   - `critic/q_value`, `actor/fb_q`

3. **第三行**: 细节分解
   - `fb/diag`, `fb/offdiag`, `fb/orth_loss`
   - `discriminator/expert_loss`, `discriminator/train_loss`

4. **第四行**: 健康检查
   - `fb/b_norm`
   - 梯度范数 (如果监测)

---

## 代码参考

### 指标收集位置

```python
# 文件: protomotions/agents/fb_cpr/agent.py

# FB Loss 指标 (第 323-410 行)
def update_fb(...) -> Dict[str, torch.Tensor]:
    metrics = {
        "fb/loss": fb_loss,
        "fb/diag": fb_diag,
        "fb/offdiag": fb_offdiag,
        "fb/orth_loss": orth_loss,
        "fb/b_norm": B.norm(dim=-1).mean(),
    }
    return metrics

# Discriminator 指标 (第 305-343 行)
def update_discriminator(...) -> Dict[str, torch.Tensor]:
    metrics = {
        "discriminator/loss": loss,
        "discriminator/expert_loss": expert_loss,
        "discriminator/train_loss": train_loss,
        "discriminator/wgan_gp": wgan_gp,
    }
    return metrics

# Critic 指标 (第 231-282 行)
def critic_step(...) -> Tuple[Tensor, Dict]:
    log_dict = {
        "losses/critic_loss": critic_loss,
        "critic/q_value": q_values_mean,
    }
    return critic_loss, log_dict

# Actor 指标 (第 194-229 行)
def actor_step(...) -> Tuple[Tensor, Dict]:
    log_dict = {
        "losses/actor_loss": actor_loss,
        "actor/fb_q": Q_fb,
        "actor/critic_q": Q_critic,
    }
    return actor_loss, log_dict
```

---

## 常见问题

### Q: 是否需要监控所有 14 个指标?

**A**:
- **最少**: 监控上面的 7 个关键指标
- **推荐**: 监控全部 14 个，特别是调参时期

### Q: 指标可以有噪声吗?

**A**: 可以，特别是:
- 早期训练 (第 1-10 epoch)
- Batch size 较小时
- 判别器和其他网络还在竞争时

关键是看总体趋势，不要看单个 epoch。

### Q: 多久看一次指标?

**A**:
- 实时监控: TensorBoard / Weights & Biases
- 每 10 epoch 人工检查一次
- 如果怀疑有问题，立即暂停检查

### Q: 是否需要调整所有超参?

**A**: 不需要。从这个优先级开始:
1. 第一优先: `embedding_layers` (质量)
2. 第二优先: 学习率 (稳定性)
3. 第三优先: `batch_size` (效率)
4. 最后: 其他系数

---

## 总结

✅ **我们完全实现了所有关键指标**，您可以:
- 直接开始训练，无需修改任何指标代码
- 自动获得 TensorBoard 可视化
- 使用本指南诊断和优化训练
- 对比不同超参的效果

**祝训练顺利！** 🚀

