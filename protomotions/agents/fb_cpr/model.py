"""
FB-CPR Model for ProtoMotions
Implements Forward-Backward representations with Conditional Policy Regularization
Matched with metamotivo architecture
"""

import dataclasses
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from hydra.utils import instantiate
import numpy as np


@dataclasses.dataclass
class MLPConfig:
    """MLP layer configuration"""
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"
    use_layer_norm: bool = False


class Norm(nn.Module):
    """Normalize vector to unit sphere"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class DenseParallel(nn.Module):
    """Parallel dense layer for multiple forward passes"""
    def __init__(self, in_features: int, out_features: int, n_parallel: int,
                 bias: bool = True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel

        self.weight = nn.Parameter(
            torch.empty((n_parallel, in_features, out_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((n_parallel, 1, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, input, self.weight)


class ParallelLayerNorm(nn.Module):
    """Parallel layer norm for multiple forward passes"""
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(n_parallel, 1, *self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(n_parallel, 1, *self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        norm_input = F.layer_norm(
            input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        return norm_input


def simple_embedding(input_dim: int, hidden_dim: int, hidden_layers: int,
                     num_parallel: int = 1) -> nn.Module:
    """Simple embedding network (matches metamotivo)"""
    assert hidden_layers >= 2, "must have at least 2 embedding layers"

    def linear(in_dim, out_dim, n_parallel=1):
        if n_parallel > 1:
            return DenseParallel(in_dim, out_dim, n_parallel=n_parallel)
        return nn.Linear(in_dim, out_dim)

    def layernorm(dim, n_parallel=1):
        if n_parallel > 1:
            return ParallelLayerNorm([dim], n_parallel=n_parallel)
        return nn.LayerNorm(dim)

    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class BackwardMap(nn.Module):
    """Backward map B: obs -> z_dim (matches metamotivo)
    Embeds observations into latent space with Tanh activation and normalization
    """
    def __init__(self, obs_dim: int, z_dim: int, hidden_dim: int = 256,
                 hidden_layers: int = 2, norm: bool = True):
        super().__init__()

        seq = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, z_dim)]
        if norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            z: (batch, z_dim) - normalized
        """
        return self.net(obs)


class ForwardMap(nn.Module):
    """Forward map F: (obs, z, action) -> z_dim (matches metamotivo)
    Predicts latent state representation after action with dual embeddings
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int,
                 hidden_dim: int = 256, hidden_layers: int = 1,
                 embedding_layers: int = 2, num_parallel: int = 2, output_dim=None):
        super().__init__()
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        # Two embedding branches
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel)

        # Main network
        def linear(in_dim, out_dim, n_parallel=1):
            if n_parallel > 1:
                return DenseParallel(in_dim, out_dim, n_parallel=n_parallel)
            return nn.Linear(in_dim, out_dim)

        seq = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        seq += [linear(hidden_dim, output_dim if output_dim else z_dim, num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
            z: (batch, z_dim)
            action: (batch, action_dim)
        Returns:
            z_pred: (num_parallel, batch, z_dim) if num_parallel > 1 else (batch, z_dim)
        """
        if self.num_parallel > 1:
            obs = obs.unsqueeze(0).expand(self.num_parallel, -1, -1)
            z = z.unsqueeze(0).expand(self.num_parallel, -1, -1)
            action = action.unsqueeze(0).expand(self.num_parallel, -1, -1)

        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class TruncatedNormal(torch.distributions.Normal):
    """Truncated normal distribution (matches metamotivo)"""
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped_x.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Actor(nn.Module):
    """Actor network: (obs, z) -> action distribution (matches metamotivo)
    Uses two embedding branches: one for (obs, z) and one for (obs)
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int,
                 hidden_dim: int = 256, hidden_layers: int = 1,
                 embedding_layers: int = 2):
        super().__init__()

        # Two embedding branches
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = simple_embedding(obs_dim, hidden_dim, embedding_layers)

        # Policy network
        def linear(in_dim, out_dim):
            return nn.Linear(in_dim, out_dim)

        seq = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        """
        Args:
            obs: (batch, obs_dim)
            z: (batch, z_dim)
            std: Standard deviation for output distribution
        Returns:
            dist: TruncatedNormal distribution
        """
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class Discriminator(nn.Module):
    """Discriminator network: (obs, z) -> logits (matches metamotivo)
    Discriminates between expert and non-expert state-latent pairs with LayerNorm + Tanh
    """
    def __init__(self, obs_dim: int, z_dim: int, hidden_dim: int = 256,
                 hidden_layers: int = 2):
        super().__init__()

        seq = [nn.Linear(obs_dim + z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Sigmoid output for probability"""
        s = self.compute_logits(obs, z)
        return torch.sigmoid(s)

    def compute_logits(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits (raw output before sigmoid)"""
        x = torch.cat([z, obs], dim=1)
        logits = self.trunk(x)
        return logits

    def compute_reward(self, obs: torch.Tensor, z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Compute implicit reward from discriminator using log-odds"""
        s = self.forward(obs, z)
        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward


class Critic(nn.Module):
    """Critic network: (obs, z, action) -> Q-value (matches metamotivo)
    Estimates value of state-action-latent tuples with parallel support
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int,
                 hidden_dim: int = 256, hidden_layers: int = 1,
                 embedding_layers: int = 2, num_parallel: int = 1, output_dim=None):
        super().__init__()
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        # Two embedding branches (one for z, one for s+a)
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel)

        # Main network
        def linear(in_dim, out_dim, n_parallel=1):
            if n_parallel > 1:
                return DenseParallel(in_dim, out_dim, n_parallel=n_parallel)
            return nn.Linear(in_dim, out_dim)

        seq = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        seq += [linear(hidden_dim, output_dim if output_dim else 1, num_parallel)]
        self.Qs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
            z: (batch, z_dim)
            action: (batch, action_dim)
        Returns:
            q_value: (num_parallel, batch, 1) if num_parallel > 1 else (batch, 1)
        """
        if self.num_parallel > 1:
            obs = obs.unsqueeze(0).expand(self.num_parallel, -1, -1)
            z = z.unsqueeze(0).expand(self.num_parallel, -1, -1)
            action = action.unsqueeze(0).expand(self.num_parallel, -1, -1)

        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        return self.Qs(torch.cat([sa_embedding, z_embedding], dim=-1))


@dataclasses.dataclass
class FBCPRModelConfig:
    """Configuration for FB-CPR model"""
    obs_dim: int = -1
    action_dim: int = -1
    z_dim: int = 64
    device: str = "cpu"
    normalize_obs: bool = True


class FBCPRModel(nn.Module):
    """FB-CPR Model combining Forward-Backward representations with CPR
    Matches metamotivo architecture with parallel networks
    """

    def __init__(self, obs_dim: int, action_dim: int, z_dim: int = 64,
                 hidden_dim: int = 1024, hidden_layers: int = 1,
                 embedding_layers: int = 2, num_parallel: int = 2,
                 normalize_obs: bool = True, device: str = "cpu",
                 actor_std: float = 0.2, **kwargs):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.device = device
        self.actor_std = actor_std

        # Create networks with metamotivo architecture
        self.forward_map = ForwardMap(
            obs_dim, z_dim, action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            embedding_layers=embedding_layers,
            num_parallel=num_parallel
        )
        self.backward_map = BackwardMap(
            obs_dim, z_dim,
            hidden_dim=256,
            hidden_layers=2,
            norm=True
        )
        self.actor = Actor(
            obs_dim, z_dim, action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            embedding_layers=embedding_layers
        )
        self.discriminator = Discriminator(
            obs_dim, z_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2
        )
        self.critic = Critic(
            obs_dim, z_dim, action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            embedding_layers=embedding_layers,
            num_parallel=num_parallel
        )

        # Create target networks
        self.target_forward_map = copy.deepcopy(self.forward_map)
        self.target_backward_map = copy.deepcopy(self.backward_map)
        self.target_critic = copy.deepcopy(self.critic)

        # Observation normalizer
        if normalize_obs:
            self.obs_normalizer = nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01)
        else:
            self.obs_normalizer = nn.Identity()

        # Initialize to eval mode, no gradients by default
        self.eval()
        self.requires_grad_(False)
        self.to(device)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations"""
        if isinstance(self.obs_normalizer, nn.Identity):
            return obs
        with torch.no_grad():
            if self.obs_normalizer.training:
                return self.obs_normalizer(obs)
            else:
                self.obs_normalizer.eval()
                return self.obs_normalizer(obs)

    def sample_z(self, batch_size: int, device: Optional[str] = None) -> torch.Tensor:
        """Sample latent variables from standard Gaussian and normalize"""
        if device is None:
            device = self.device
        z = torch.randn(batch_size, self.z_dim, dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project z to unit sphere (normalization)"""
        return math.sqrt(self.z_dim) * F.normalize(z, dim=-1)

    def to(self, *args, **kwargs):
        """Override to method to update device"""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device.type if hasattr(device, 'type') else str(device)
        return super().to(*args, **kwargs)
