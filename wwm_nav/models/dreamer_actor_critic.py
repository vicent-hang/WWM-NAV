from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class DreamerImagination:
    features: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    pred_observations: torch.Tensor
    pred_outages: torch.Tensor
    done_probs: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    weights: torch.Tensor
    actor_probs: torch.Tensor | None = None
    nav_prior_probs: torch.Tensor | None = None
    reward_components: dict[str, torch.Tensor] | None = None


def build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ELU())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class DreamerV3DiscreteActor(nn.Module):
    """DreamerV3-style categorical actor with unimix and small output scale."""

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (400, 400, 400, 400),
        unimix: float = 0.01,
        outscale: float = 0.01,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.unimix = float(unimix)
        self.net = build_mlp(feat_dim, action_dim, hidden_dims)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            final.weight.data.mul_(float(outscale))
            final.bias.data.mul_(float(outscale))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())

    def distribution(self, features: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Categorical:
        logits = self(features)
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
        empty_mask = valid_mask.sum(dim=-1, keepdim=True) == 0
        if bool(torch.any(empty_mask).detach().cpu()):
            valid_mask = torch.where(empty_mask, torch.ones_like(valid_mask, dtype=torch.bool), valid_mask)
        probs = torch.softmax(torch.where(valid_mask, logits, torch.full_like(logits, -1.0e9)), dim=-1)
        valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        uniform = valid_mask.to(probs.dtype) / valid_count.to(probs.dtype)
        if self.unimix > 0.0:
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        probs = torch.where(valid_mask, probs.clamp_min(1.0e-8), torch.zeros_like(probs))
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return Categorical(probs=probs)

    def act(self, features: torch.Tensor, deterministic: bool = False, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        dist = self.distribution(features, valid_mask=valid_mask)
        if deterministic:
            return torch.argmax(dist.probs, dim=-1)
        return dist.sample()


class DreamerCritic(nn.Module):
    """Scalar value critic for latent RSSM features."""

    def __init__(self, feat_dim: int, hidden_dims: tuple[int, ...] = (400, 400, 400, 400)) -> None:
        super().__init__()
        self.net = build_mlp(feat_dim, 1, hidden_dims)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)


class EMAScalarNormalizer(nn.Module):
    """Small EMA scale tracker used for Dreamer-style return/advantage scaling."""

    def __init__(self, rate: float = 0.01, min_scale: float = 1.0e-4) -> None:
        super().__init__()
        self.rate = float(rate)
        self.min_scale = float(min_scale)
        self.register_buffer("offset", torch.zeros(()))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        finite = values.detach()[torch.isfinite(values.detach())]
        if finite.numel() == 0:
            return
        offset = finite.mean()
        scale = finite.std(unbiased=False).clamp_min(self.min_scale)
        if not bool(self.initialized.detach().cpu()):
            self.offset.copy_(offset)
            self.scale.copy_(scale)
            self.initialized.fill_(True)
            return
        self.offset.mul_(1.0 - self.rate).add_(offset, alpha=self.rate)
        self.scale.mul_(1.0 - self.rate).add_(scale, alpha=self.rate)
        self.scale.clamp_(min=self.min_scale)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.offset.to(values.device)) / self.scale.to(values.device).clamp_min(self.min_scale)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale.to(values.device).clamp_min(self.min_scale) + self.offset.to(values.device)


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Dreamer-style lambda return.

    Args:
        rewards: [H, B] rewards for imagined transitions.
        values: [H, B] values for states s_0 ... s_{H-1}.
        discounts: [H, B] continuation discounts for each transition.
        bootstrap: [B] value for the final state s_H.
        lambda_: scalar trace parameter.
    """

    if rewards.shape != values.shape or rewards.shape != discounts.shape:
        raise ValueError(
            "rewards, values, and discounts must have the same shape; "
            f"got {rewards.shape}, {values.shape}, {discounts.shape}"
        )
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    inputs = rewards + discounts * next_values * (1.0 - float(lambda_))
    returns = []
    accumulator = bootstrap
    for index in reversed(range(rewards.shape[0])):
        accumulator = inputs[index] + discounts[index] * float(lambda_) * accumulator
        returns.append(accumulator)
    returns.reverse()
    return torch.stack(returns, dim=0)


def update_slow_target(source: nn.Module, target: nn.Module, fraction: float) -> None:
    """Dreamer-style slow target update.

    fraction=1.0 copies source into target. Smaller values perform Polyak
    averaging: target <- fraction * source + (1 - fraction) * target.
    """

    mix = float(fraction)
    with torch.no_grad():
        for source_param, target_param in zip(
            source.parameters(), target.parameters(), strict=True
        ):
            target_param.data.mul_(1.0 - mix).add_(source_param.data, alpha=mix)
