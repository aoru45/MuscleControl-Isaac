# FB-CPR Agent for ProtoMotions

Forward-Backward representations with Conditional Policy Regularization (FB-CPR) implementation for ProtoMotions framework.

完整匹配 metamotivo 网络架构，包括并行网络、双 embedding 分支和截断正态分布。

## Overview

FB-CPR combines two key concepts:
1. **Forward-Backward (FB) Representations**: Learns latent representations through forward (state transition) and backward (state embedding) maps
2. **Conditional Policy Regularization (CPR)**: Uses a discriminator to regularize the policy towards expert demonstrations and a critic for value estimation

## Architecture

### Networks (Matches Metamotivo)

The implementation includes five main neural networks:

#### 1. Forward Map F: `(obs, z, action) → z'`
   - **Purpose**: Predicts next latent representation after action
   - **Architecture**:
     - Dual embedding branches: `embed_z(obs+z)` and `embed_sa(obs+action)`
     - Each branch: `simple_embedding` with `LayerNorm + Tanh + ReLU layers`
     - Output dimension: z_dim
     - **Parallel support**: Multiple forward heads via `DenseParallel` layers for uncertainty estimation
     - Output shape: `(num_parallel, batch, z_dim)`

#### 2. Backward Map B: `obs → z`
   - **Purpose**: Embeds observations into latent space (used for goal encoding)
   - **Architecture**:
     - `LayerNorm + Tanh` first layer
     - ReLU hidden layers
     - Normalized output via `Norm()` layer (unit sphere: `√z_dim * normalize(z)`)
   - Output shape: `(batch, z_dim)` with unit norm

#### 3. Actor: `(obs, z) → TruncatedNormal`
   - **Purpose**: Policy network generating actions conditioned on observation and latent code
   - **Architecture**:
     - Dual embedding branches: `embed_z(obs+z)` and `embed_s(obs)`
     - Each branch outputs: `hidden_dim // 2`
     - Concatenated embeddings → policy MLP → mean (with tanh activation)
     - Returns `TruncatedNormal` distribution with std parameter
   - Output: Action distribution in `[-1, 1]` range

#### 4. Discriminator: `(obs, z) → logits`
   - **Purpose**: Discriminates between expert and training data (state-latent pairs)
   - **Architecture**:
     - `LayerNorm + Tanh` first layer (stable initialization)
     - ReLU hidden layers
     - Output layer: single scalar logit
     - **Reward computation**: log-odds ratio `log(sigmoid(logits)) - log(1-sigmoid(logits))`
   - Uses WGAN-GP for stable training
   - Output shape: `(batch, 1)`

#### 5. Critic: `(obs, z, action) → Q-value`
   - **Purpose**: Estimates value of state-action-latent tuples
   - **Architecture**:
     - Dual embedding branches: `embed_z(obs+z)` and `embed_sa(obs+action)`
     - Similar to Forward Map
     - **Parallel support**: Via `DenseParallel` layers (matches forward map parallelism)
   - Output shape: `(num_parallel, batch, 1)`

## Goal Encoding 详解

**Goal encoding** 是将目标状态（通常是 next_obs）通过 Backward map 编码到潜在空间的过程：

```python
# Goal encoding 计算
goals = backward_map(next_obs)  # (batch, z_dim)
goals = project_z(goals)         # 归一化到单位超球面
```

### Goal Encoding 的作用

1. **目标状态表示**：
   - 通过 Backward map 将观测空间的目标状态映射到潜在空间
   - 得到的 z 代表"应该达到什么目标"

2. **促进策略学习**：
   - 采样时混合使用：goal encoding 的 z（鼓励趋向目标）+ 随机 z + 专家 z
   - 帮助策略学会到达特定目标状态

3. **Z 采样混合**：
   ```python
   # Mixed z distribution
   p_goal = 0.5        # 50% 来自 goal encoding
   p_expert = 0.0      # 0% 来自专家编码
   p_random = 0.5      # 50% 来自均匀分布

   z_uniform = sample_z()                          # 高斯采样
   z_goal = backward_map(next_obs)                # Goal encoding
   z_expert = backward_map(expert_next_obs)       # Expert encoding

   # 根据概率混合
   z = mix(z_goal, z_expert, z_uniform)
   ```

### 关键特点

- **自适应探索**：通过目标编码引导策略探索
- **多样性保证**：混合多种 z 来源避免模式坍塌
- **无需手工设计目标**：从数据中自动学习有意义的目标表示

---

### Training Objectives

1. **Forward-Backward Loss**: Matches latent space transitions between forward and backward maps
   - Diagonal matching: Encourages consistency of z representations
   - Off-diagonal minimization: Prevents trivial solutions
   - Orthonormality constraint: Maintains orthonormal basis in latent space

2. **Discriminator Loss**: Binary cross-entropy with gradient penalty
   - Trains discriminator to distinguish expert from training data
   - WGAN gradient penalty prevents mode collapse

3. **Critic Loss**: Temporal difference learning
   - Estimates value using discriminator reward signal
   - Soft target network for stability

4. **Actor Loss**: Policy gradient with two components
   - Maximizes discriminator rewards (policy regularization)
   - Maximizes forward map Q-values (latent space exploration)

---

## 并行网络 (Parallel Networks) 设计

为了改进值估计的鲁棒性，FB-CPR 使用并行网络结构：

### 并行前向映射和评论家

```
Forward Map / Critic Architecture (num_parallel=2):

Batch Input: obs, z, action
     ↓
DenseParallel Linear Layer (2 independent heads)
     ↓
[Head 0]  [Head 1]    (2 parallel computations)
     ↓         ↓
Output: (2, batch_size, output_dim)
```

### 优势

1. **不确定性估计**：多头网络的差异反映预测不确定性
2. **鲁棒性**：通过平均化多个预测减少方差
3. **悲观性**：使用最小值或平均值 - ζ*方差 进行保守估计

### 参数配置

```yaml
num_parallel: 2              # 并行网络数量
ensemble_mode: "batch"       # 批处理模式（推荐）
```

---

## Usage

### Configuration

The FB-CPR agent can be configured via YAML files in `/protomotions/config/agent/fb_cpr/`.

Basic configuration parameters:

```yaml
agent:
  _target_: protomotions.agents.fb_cpr.agent.FBCPRAgent
  config:
    model:
      z_dim: 64  # Latent dimension
      normalize_obs: true

    fbcpr_config:
      # Learning rates
      lr_forward: 1e-4
      lr_backward: 1e-4
      lr_critic: 1e-4
      lr_discriminator: 1e-4
      lr_actor: 1e-4

      # Loss coefficients
      ortho_coef: 1.0
      reg_coeff: 1.0
      grad_penalty_discriminator: 10.0

      # Soft update coefficients
      fb_target_tau: 0.005
      critic_target_tau: 0.005

      # Z sampling
      train_goal_ratio: 0.5
      expert_asm_ratio: 0.0

      # Training
      batch_size: 256
      discount: 0.99
```

### Training

Example usage with ProtoMotions:

```python
from lightning.fabric import Fabric
from protomotions.agents.fb_cpr import FBCPRAgent
from hydra import compose, initialize_config_dir

# Load configuration
config = compose(config_name="fb_cpr_config")

# Initialize fabric for distributed training
fabric = Fabric(devices=4, strategy="ddp")

# Create environment and agent
env = create_env(config.env_config)
agent = FBCPRAgent(fabric=fabric, env=env, config=config.agent.config)
agent.setup()

# Train
agent.fit()
```

## Files

- `model.py`: Network architectures and FBCPRModel
- `agent.py`: FBCPRAgent training logic
- `__init__.py`: Package initialization
- `../config/agent/fb_cpr/agent.yaml`: Main configuration
- `../config/agent/fb_cpr/models/`: Individual network configs

## Implementation Details

### Key Differences from Metamotivo

1. **Simplified Network Architecture**: Uses simple MLPs instead of complex ensemble networks
2. **ProtoMotions Integration**: Extends PPO base class for compatibility
3. **Experience Buffer**: Integrates with ProtoMotions' experience buffer system
4. **Configuration**: Uses Hydra YAML configuration instead of dataclasses

### Main Training Loop

1. **Collect Experience**: Actor explores environment using sampled z
2. **Sample Mixed Z**: Mix z from three distributions:
   - Goal encodings from backward map (train_goal_ratio)
   - Expert encodings (expert_asm_ratio)
   - Uniform samples (remaining)
3. **Update Discriminator**: Train to distinguish expert from training data
4. **Update FB Networks**: Optimize forward and backward maps for consistency
5. **Update Critic**: Learn value function using discriminator rewards
6. **Update Actor**: Maximize both discriminator and forward map Q-values
7. **Soft Update**: Apply target network updates

## Parameters Guide

- **z_dim**: Latent dimension (higher = more capacity)
- **ortho_coef**: Strength of orthonormality constraint (higher = stricter)
- **reg_coeff**: Strength of policy regularization (higher = follow expert more)
- **train_goal_ratio**: How much to use goal encodings for z sampling
- **grad_penalty_discriminator**: WGAN-GP weight (higher = more stable but slower)
- **critic_target_tau**: Soft update speed for target networks (higher = faster)

## Troubleshooting

1. **Training instability**: Reduce grad_penalty_discriminator or increase critic_target_tau
2. **Poor discriminator performance**: Increase lr_discriminator or grad_penalty_discriminator
3. **Slow convergence**: Increase learning rates or reduce ortho_coef
4. **Divergence**: Check batch_size and reduce learning rates

## References

- Forward-Backward Representations: From the metamotivo paper
- WGAN-GP: Improved Training of Wasserstein GANs
- Soft Actor-Critic: Offline RL framework adapted for our use case
