"""
FB-CPR Agent for ProtoMotions
Implements training logic for Forward-Backward representations with Conditional Policy Regularization
"""

import dataclasses
import torch
import torch.nn.functional as F
from torch import autograd
import logging
from typing import Dict, Optional, Tuple

from lightning.fabric import Fabric
from hydra.utils import instantiate

from protomotions.agents.ppo.agent import PPO
from protomotions.agents.fb_cpr.model import FBCPRModel
from protomotions.envs.base_env.env import BaseEnv
from protomotions.utils.running_mean_std import RunningMeanStd

log = logging.getLogger(__name__)


@dataclasses.dataclass
class FBCPRTrainConfig:
    """FB-CPR specific training configuration"""
    # Learning rates
    lr_forward: float = 1e-4
    lr_backward: float = 1e-4
    lr_critic: float = 1e-4
    lr_discriminator: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0

    # Network update parameters
    fb_target_tau: float = 0.005  # Soft update coefficient for FB networks
    critic_target_tau: float = 0.005  # Soft update coefficient for critic

    # FB loss parameters
    ortho_coef: float = 1.0  # Orthonormality loss coefficient
    fb_pessimism_penalty: float = 0.5

    # Critic loss parameters
    critic_pessimism_penalty: float = 0.5

    # Actor loss parameters
    actor_pessimism_penalty: float = 0.5
    reg_coeff: float = 1.0  # Regulation coefficient for actor loss
    scale_reg: bool = True  # Whether to scale regulation by Q value

    # Discriminator parameters
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0

    # Z sampling parameters
    expert_asm_ratio: float = 0.0  # Ratio of expert encoded z
    train_goal_ratio: float = 0.5  # Ratio of goal encoded z
    relabel_ratio: Optional[float] = None  # Probability of z relabeling

    # Other parameters
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    clip_grad_norm: float = 0.0
    batch_size: int = 256
    discount: float = 0.99


class FBCPRAgent(PPO):
    """FB-CPR Agent extending PPO with Forward-Backward and CPR"""

    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        # Extract FB-CPR specific config
        self.fbcpr_config = FBCPRTrainConfig()
        if hasattr(config, 'fbcpr_config'):
            # Update with provided config
            for key, value in dataclasses.asdict(self.fbcpr_config).items():
                if hasattr(config.fbcpr_config, key):
                    setattr(self.fbcpr_config, key, getattr(config.fbcpr_config, key))

        super().__init__(fabric, env, config)

        self.discriminator_optimizer = None
        self.critic_optimizer = None
        self.forward_optimizer = None
        self.backward_optimizer = None
        self.actor_optimizer_fb = None

    def setup(self):
        """Setup FB-CPR model and optimizers"""
        # Create FB-CPR model
        model: FBCPRModel = instantiate(
            self.config.model,
            obs_dim=self.env.config.robot.self_obs_size,
            action_dim=self.env.config.robot.number_of_actions,
        )
        model = model.to(self.device)

        # Create optimizers for each network
        self.forward_optimizer = torch.optim.Adam(
            model.forward_map.parameters(),
            lr=self.fbcpr_config.lr_forward,
            weight_decay=self.fbcpr_config.weight_decay,
        )
        self.backward_optimizer = torch.optim.Adam(
            model.backward_map.parameters(),
            lr=self.fbcpr_config.lr_backward,
            weight_decay=self.fbcpr_config.weight_decay,
        )
        self.actor_optimizer_fb = torch.optim.Adam(
            model.actor.parameters(),
            lr=self.fbcpr_config.lr_actor,
            weight_decay=self.fbcpr_config.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            model.critic.parameters(),
            lr=self.fbcpr_config.lr_critic,
            weight_decay=self.fbcpr_config.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=self.fbcpr_config.lr_discriminator,
            weight_decay=self.fbcpr_config.weight_decay_discriminator,
        )

        # Setup with fabric
        self.model, *optimizers = self.fabric.setup(
            model,
            self.forward_optimizer,
            self.backward_optimizer,
            self.actor_optimizer_fb,
            self.critic_optimizer,
            self.discriminator_optimizer,
        )
        (self.forward_optimizer, self.backward_optimizer, self.actor_optimizer_fb,
         self.critic_optimizer, self.discriminator_optimizer) = optimizers

        # Setup observation normalizer if needed
        if self.fbcpr_config.lr_actor > 0:
            self.model.obs_normalizer.train()

        # For compatibility with PPO methods
        self.actor_optimizer = self.actor_optimizer_fb
        self.critic_optimizer_ppo = None  # FB-CPR uses its own critic

    def register_extra_experience_buffer_keys(self):
        """Register FB-CPR specific buffer keys"""
        # Register z latent variables
        self.experience_buffer.register_key(
            "z", shape=(self.fbcpr_config.batch_size,)
        )

    @torch.no_grad()
    def sample_mixed_z(self, train_goals: torch.Tensor,
                       expert_encodings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample z from mixed distribution"""
        batch_size = train_goals.shape[0]
        z = self.model.sample_z(batch_size)

        # Mix z with goal encodings
        p_goal = self.fbcpr_config.train_goal_ratio
        p_expert_asm = self.fbcpr_config.expert_asm_ratio
        prob = torch.tensor(
            [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
            dtype=torch.float32,
            device=self.device,
        )
        mix_idxs = torch.multinomial(prob, num_samples=batch_size, replacement=True).reshape(-1, 1)

        # Goals from train trajectory encoding
        perm = torch.randperm(batch_size, device=self.device)
        goals = self.model.backward_map(train_goals[perm])
        goals = self.model.project_z(goals)
        z = torch.where(mix_idxs == 0, goals, z)

        # Expert encodings if available
        if expert_encodings is not None and p_expert_asm > 0:
            perm = torch.randperm(batch_size, device=self.device)
            z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    def act(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None,
            mean: bool = True) -> torch.Tensor:
        """Get action from policy"""
        if z is None:
            z = self.model.sample_z(obs.shape[0])

        obs_norm = self.model._normalize_obs(obs)
        dist = self.model.actor(obs_norm, z, self.model.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    def actor_step(self, batch_dict) -> Tuple[torch.Tensor, Dict]:
        """FB-CPR actor update step"""
        obs = batch_dict["self_obs"]
        z = batch_dict.get("z")
        if z is None:
            z = self.model.sample_z(obs.shape[0])

        # Normalize obs
        obs_norm = self.model._normalize_obs(obs)

        # Get action distribution and sample
        dist = self.model.actor(obs_norm, z, self.model.actor_std)
        action = dist.sample(clip=self.fbcpr_config.stddev_clip)

        # Get forward predictions for Q estimation
        Fs = self.model.forward_map(obs_norm, z, action)
        Qs_fb = (Fs * z).sum(-1)  # (num_parallel, batch) -> take mean

        # Estimate Q value with pessimism
        Q_mean = Qs_fb.mean(dim=0)  # Average over parallel networks
        Q_fb = Q_mean

        # Get critic Q value
        Qs_critic = self.model.critic(obs_norm, z, action)  # (num_parallel, batch, 1)
        Q_critic = Qs_critic.mean(dim=0).squeeze(-1)  # Average over parallel networks

        # Actor loss: maximize both FB and critic Q values
        weight = Q_fb.abs().detach() if self.fbcpr_config.scale_reg else 1.0
        actor_loss = -Q_critic - self.fbcpr_config.reg_coeff * weight * Q_fb

        log_dict = {
            "actor/fb_q": Q_fb.detach().mean(),
            "actor/critic_q": Q_critic.detach().mean(),
            "losses/actor_loss": actor_loss.detach().mean(),
        }
        return actor_loss.mean(), log_dict

    def critic_step(self, batch_dict) -> Tuple[torch.Tensor, Dict]:
        """FB-CPR critic update step"""
        obs = batch_dict["self_obs"]
        action = batch_dict.get("actions")
        reward = batch_dict.get("rewards")
        next_obs = batch_dict.get("next_obs")
        if next_obs is None and "next" in batch_dict and "observation" in batch_dict["next"]:
            next_obs = batch_dict["next"]["observation"]

        z = batch_dict.get("z")
        if z is None:
            z = self.model.sample_z(obs.shape[0])

        obs_norm = self.model._normalize_obs(obs)
        if next_obs is not None:
            next_obs_norm = self.model._normalize_obs(next_obs)
        else:
            next_obs_norm = None

        if action is None or reward is None or next_obs_norm is None:
            # Fallback: compute loss on discriminator reward
            disc_reward = self.model.discriminator.compute_reward(obs_norm, z)
            target_q = disc_reward.unsqueeze(-1)
        else:
            # Compute target Q value with discriminator reward
            with torch.no_grad():
                dist = self.model.actor(next_obs_norm, z, self.model.actor_std)
                next_action = dist.sample(clip=self.fbcpr_config.stddev_clip)

                # Get discriminator reward
                disc_reward = self.model.discriminator.compute_reward(obs_norm, z)

                # Get next Q value from target critic
                target_q_next = self.model.target_critic(next_obs_norm, z, next_action)
                # Average over parallel networks
                target_q_next = target_q_next.mean(dim=0)

                # TD target
                discount = self.fbcpr_config.discount * ~batch_dict.get("dones", torch.zeros(obs.shape[0], dtype=torch.bool, device=self.device))
                target_q = disc_reward.unsqueeze(-1) + discount.unsqueeze(-1).float() * target_q_next

        # Compute critic loss
        q_values = self.model.critic(obs_norm, z, action if action is not None else torch.zeros(obs.shape[0], self.env.config.robot.number_of_actions, device=self.device))
        # Average over parallel networks
        q_values_mean = q_values.mean(dim=0)
        critic_loss = F.mse_loss(q_values_mean, target_q)

        log_dict = {
            "losses/critic_loss": critic_loss.detach(),
            "critic/q_value": q_values_mean.mean().detach(),
        }
        return critic_loss, log_dict

    @torch.no_grad()
    def gradient_penalty_wgan(self, real_obs: torch.Tensor, real_z: torch.Tensor,
                              fake_obs: torch.Tensor, fake_z: torch.Tensor) -> torch.Tensor:
        """Compute WGAN gradient penalty for discriminator"""
        batch_size = real_obs.shape[0]
        alpha = torch.rand(batch_size, 1, device=self.device)

        interpolates = torch.cat([
            (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True),
            (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True),
        ], dim=1)

        d_interpolates = self.model.discriminator(
            interpolates[:, :real_obs.shape[1]],
            interpolates[:, real_obs.shape[1]:]
        )

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_discriminator(self, expert_obs: torch.Tensor, expert_z: torch.Tensor,
                             train_obs: torch.Tensor, train_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update discriminator network"""
        # Discriminator loss: binary cross entropy for real vs fake
        expert_logits = self.model.discriminator(expert_obs, expert_z)
        train_logits = self.model.discriminator(train_obs, train_z)

        expert_loss = -F.logsigmoid(expert_logits).mean()
        train_loss = F.softplus(train_logits).mean()
        loss = expert_loss + train_loss

        # WGAN gradient penalty
        if self.fbcpr_config.grad_penalty_discriminator > 0:
            wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            loss = loss + self.fbcpr_config.grad_penalty_discriminator * wgan_gp
        else:
            wgan_gp = torch.tensor(0.0, device=self.device)

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()

        metrics = {
            "discriminator/loss": loss.detach(),
            "discriminator/expert_loss": expert_loss.detach(),
            "discriminator/train_loss": train_loss.detach(),
        }
        if self.fbcpr_config.grad_penalty_discriminator > 0:
            metrics["discriminator/wgan_gp"] = wgan_gp.detach()

        return metrics

    def update_fb(self, obs: torch.Tensor, action: torch.Tensor,
                  next_obs: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update forward and backward maps"""
        # Normalize
        obs_norm = self.model._normalize_obs(obs)
        next_obs_norm = self.model._normalize_obs(next_obs)

        # Get target predictions
        with torch.no_grad():
            dist = self.model.actor(next_obs_norm, z, self.model.actor_std)
            next_action = dist.sample(clip=self.fbcpr_config.stddev_clip)

            target_Fs = self.model.target_forward_map(next_obs_norm, z, next_action)
            target_B = self.model.target_backward_map(next_obs_norm)

            # M matrix: representation matching
            # target_Fs: (num_parallel, batch, z_dim)
            # target_B: (batch, z_dim)
            # Ms: (num_parallel, batch, batch)
            target_Ms = torch.matmul(target_Fs, target_B.T)
            target_M = target_Ms.mean(dim=0)  # Average over parallel networks

        # Current predictions
        Fs = self.model.forward_map(obs_norm, z, action)
        B = self.model.backward_map(obs_norm)
        Ms = torch.matmul(Fs, B.T)
        M = Ms.mean(dim=0)  # Average over parallel networks

        # FB loss: temporal difference on representation matching
        batch_size = obs.shape[0]
        off_diag = 1 - torch.eye(batch_size, device=self.device)

        diff = M - target_M
        fb_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag.sum()
        fb_diag = -torch.diagonal(diff).mean()
        fb_loss = fb_offdiag + fb_diag

        # Orthonormality loss for B
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * off_diag).pow(2).sum() / off_diag.sum()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss = fb_loss + self.fbcpr_config.ortho_coef * orth_loss

        # Optimize FB networks
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()

        if self.fbcpr_config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.forward_map.parameters(),
                                          self.fbcpr_config.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.backward_map.parameters(),
                                          self.fbcpr_config.clip_grad_norm)

        self.forward_optimizer.step()
        self.backward_optimizer.step()

        metrics = {
            "fb/loss": fb_loss.detach(),
            "fb/diag": fb_diag.detach(),
            "fb/offdiag": fb_offdiag.detach(),
            "fb/orth_loss": orth_loss.detach(),
            "fb/b_norm": B.norm(dim=-1).mean().detach(),
        }
        return metrics

    def soft_update_networks(self):
        """Soft update target networks"""
        for param, target_param in zip(self.model.forward_map.parameters(),
                                      self.model.target_forward_map.parameters()):
            target_param.data.copy_(
                self.fbcpr_config.fb_target_tau * param.data +
                (1 - self.fbcpr_config.fb_target_tau) * target_param.data
            )

        for param, target_param in zip(self.model.backward_map.parameters(),
                                      self.model.target_backward_map.parameters()):
            target_param.data.copy_(
                self.fbcpr_config.fb_target_tau * param.data +
                (1 - self.fbcpr_config.fb_target_tau) * target_param.data
            )

        for param, target_param in zip(self.model.critic.parameters(),
                                      self.model.target_critic.parameters()):
            target_param.data.copy_(
                self.fbcpr_config.critic_target_tau * param.data +
                (1 - self.fbcpr_config.critic_target_tau) * target_param.data
            )

    def extra_optimization_steps(self, batch_dict, batch_idx: int) -> Dict:
        """Additional optimization steps for FB and discriminator"""
        obs = batch_dict["self_obs"]
        z = batch_dict.get("z")
        if z is None:
            z = self.model.sample_z(obs.shape[0])

        metrics = {}

        # Update FB networks
        if "next_obs" in batch_dict:
            next_obs = batch_dict.get("next_obs")
            if next_obs is None and "next" in batch_dict and "observation" in batch_dict["next"]:
                next_obs = batch_dict["next"]["observation"]

            if next_obs is not None:
                action = batch_dict.get("actions")
                if action is None:
                    # Sample action from actor
                    obs_norm = self.model._normalize_obs(obs)
                    dist = self.model.actor(obs_norm, z, self.model.actor_std)
                    action = dist.sample(clip=self.fbcpr_config.stddev_clip)

                metrics.update(self.update_fb(obs, action, next_obs, z))

        # Update discriminator if we have expert data
        if "expert_obs" in batch_dict:
            expert_obs = batch_dict["expert_obs"]
            expert_z = batch_dict.get("expert_z", self.model.sample_z(expert_obs.shape[0]))
            train_z = z
            metrics.update(
                self.update_discriminator(expert_obs, expert_z, obs, train_z)
            )

        # Soft update target networks
        self.soft_update_networks()

        return metrics

    def calculate_extra_actor_loss(self, batch_dict, dist) -> Tuple[torch.Tensor, Dict]:
        """FB-CPR already handles actor loss in actor_step, no extra loss needed"""
        return torch.tensor(0.0, device=self.device), {}
