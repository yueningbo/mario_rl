"""Small MLP actor-critic."""

from __future__ import annotations

import torch
from torch import nn

from .features import NUM_ACTIONS, STACKED_DIM


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STACKED_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, NUM_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        for module in self.body:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        return self.actor(features), self.critic(features).squeeze(-1)
