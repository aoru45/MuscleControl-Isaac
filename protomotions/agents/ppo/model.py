import torch
from torch import distributions, nn
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MultiHeadedMLP


class SimpleStudentMLP(nn.Module):
    """Simple MLP for student network to predict muscle activations"""
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.logstd = nn.Parameter(
            torch.ones(output_size) * -1.0,
            requires_grad=False,
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("features")
        mu = self.net(x)
        mu = torch.sigmoid(mu)  # Output in [0, 1]
        std = torch.exp(self.logstd)
        return distributions.Normal(mu, std)


class PPOActor(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.logstd = nn.Parameter(
            torch.ones(num_out) * config.actor_logstd,
            requires_grad=False,
        )
        self.mu: MultiHeadedMLP = instantiate(self.config.mu_model, num_out=num_out)
        self.is_student = getattr(self.config, "is_student", False)

    def forward(self, input_dict):
        mu = self.mu(input_dict)
        mu = torch.tanh(mu)
        if self.is_student:
            mu = torch.relu(mu)
        std = torch.exp(self.logstd)
        dist = distributions.Normal(mu, std)
        return dist


class PPOModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create networks
        self._actor: PPOActor = instantiate(
            self.config.actor,
        )
        self._critic: MultiHeadedMLP = instantiate(
            self.config.critic,
        )

    def get_action_and_value(self, input_dict: dict):
        dist = self._actor(input_dict)
        action = dist.sample()
        value = self._critic(input_dict).flatten()

        logstd = self._actor.logstd
        std = torch.exp(logstd)

        neglogp = self.neglogp(action, dist.mean, std, logstd)
        return action, neglogp, value.flatten()

    def act(self, input_dict: dict, mean: bool = True) -> torch.Tensor:
        dist = self._actor(input_dict)
        if mean:
            return dist.mean
        return dist.sample()

    @staticmethod
    def neglogp(x, mean, std, logstd):
        dist = distributions.Normal(mean, std)
        return -dist.log_prob(x).sum(dim=-1)

class PPOModelDual(PPOModel):
    def __init__(self, config):
        super().__init__(config)
        student_num_out = int(getattr(self.config, "student_num_out", 284))

        # Create simple MLP student network
        # Input: top-3 JtA values per DOF (50 * 3 = 150) + teacher torque (50) = 200
        self._student = SimpleStudentMLP(input_size=200, output_size=student_num_out, hidden_size=256)

    def student_act(self, input_dict: dict, mean: bool = True) -> torch.Tensor:
        dist = self._student(input_dict)
        if mean:
            return dist.mean
        return dist.sample()

    def student_distribution(self, input_dict: dict):
        return self._student(input_dict)
