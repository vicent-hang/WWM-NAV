from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            previous_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.output_dim = previous_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class DuelingQNetwork(nn.Module):
    """Legacy WWM-NAV Q network retained to preserve model-init RNG ordering."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.encoder = MLP(observation_dim, hidden_dims)
        self.value = nn.Linear(self.encoder.output_dim, 1)
        self.advantage = nn.Linear(self.encoder.output_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.encoder(observations)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
