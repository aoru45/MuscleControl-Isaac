from typing import Optional, Tuple
import platform

import torch
from torch import Tensor, nn


class RunningMeanStd(nn.Module):
    def __init__(
        self,
        epsilon: int = 1e-5,
        shape: Tuple[int, ...] = (),
        device="cuda:0",
        clamp_value: Optional[float] = None,
    ):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__()
        self.epsilon = epsilon
        tc_float = torch.float if platform.system() == "Darwin" else torch.float64
        self.register_buffer(
            "mean", torch.zeros(shape, dtype=tc_float, device=device)
        )
        self.register_buffer(
            "var", torch.ones(shape, dtype=tc_float, device=device)
        )
        # self.count = epsilon
        self.register_buffer("count", torch.ones((), dtype=torch.long, device=device))
        self.clamp_value = clamp_value

    @torch.no_grad()
    def update(self, arr: torch.tensor) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Distributed update: aggregate sums and counts
            batch_count = arr.shape[0]
            
            # Ensure tensors are on the correct device for all_reduce
            if arr.device.type != 'cpu':
                # Calculation using sum of squares
                batch_sum = torch.sum(arr, dim=0)
                batch_sq_sum = torch.sum(arr ** 2, dim=0)
                batch_count_tensor = torch.tensor(batch_count, device=arr.device, dtype=torch.float32)

                torch.distributed.all_reduce(batch_sum)
                torch.distributed.all_reduce(batch_sq_sum)
                torch.distributed.all_reduce(batch_count_tensor)

                global_count = int(batch_count_tensor.item())
                batch_mean = batch_sum / global_count
                batch_var = (batch_sq_sum / global_count) - (batch_mean ** 2)
                batch_var = torch.relu(batch_var) # numerical stability
                
                # Squeeze if necessary
                if self.mean.ndim == 0:
                    batch_mean = batch_mean.squeeze()
                    batch_var = batch_var.squeeze()
                    
                self.update_from_moments(batch_mean, batch_var, global_count)
                return

        # Local update fallback
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]

        # Squeeze batch_mean and batch_var to match self.mean and self.var shape
        if self.mean.ndim == 0:  # scalar shape
            batch_mean = batch_mean.squeeze()
            batch_var = batch_var.squeeze()

        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(
        self, batch_mean: torch.tensor, batch_var: torch.tensor, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        new_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        # Use copy_() to handle both scalar and non-scalar tensors
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.fill_(new_count)

    def maybe_clamp(self, x: Tensor):
        if self.clamp_value is None:
            return x
        else:
            return torch.clamp(x, -self.clamp_value, self.clamp_value)

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        if not un_norm:
            result = (arr - self.mean.float()) / torch.sqrt(
                self.var.float() + self.epsilon
            )
            result = self.maybe_clamp(result)
        else:
            arr = self.maybe_clamp(arr)
            result = (
                arr * torch.sqrt(self.var.float() + self.epsilon) + self.mean.float()
            )

        return result


class RewardRunningMeanStd(RunningMeanStd):
    def __init__(
        self,
        fabric,
        shape: Tuple[int, ...],
        gamma: float,
        epsilon: float = 1e-5,
        clamp_value: float = 5.0,
        device: str = "cuda:0",
    ):
        super().__init__(epsilon=epsilon, shape=shape, device=device, clamp_value=clamp_value)
        self.fabric = fabric
        self.gamma = gamma
        self.discounted_reward = None

    @torch.no_grad()
    def record_reward(self, reward: torch.tensor, terminated: torch.tensor) -> torch.tensor:
        if self.discounted_reward is None:
            self.discounted_reward = reward.clone()
        else:
            self.discounted_reward = self.discounted_reward * self.gamma * (1 - terminated.float()) + reward.clone()
        arr = self.discounted_reward
        if hasattr(self, "fabric") and self.fabric is not None:
            gathered = self.fabric.all_gather(arr)
            merged = gathered.reshape(-1, *arr.shape[1:])
            batch_mean = merged.mean(dim=0)
            batch_var = merged.var(dim=0, unbiased=False)
            batch_count = merged.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            self.update(arr)

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        if not un_norm:
            result = arr / torch.sqrt(self.var.float() + self.epsilon)
            result = self.maybe_clamp(result)
        else:
            arr = self.maybe_clamp(arr)
            result = arr * torch.sqrt(self.var.float() + self.epsilon)
        return result
