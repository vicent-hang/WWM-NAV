from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from wwm_nav.models.rssm import RSSMWorldModel


@dataclass(frozen=True)
class DreamerActorCriticConfig:
    feat_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (400, 400, 400, 400)
    discount: float = 0.99
    lambda_: float = 0.95
    actor_entropy: float = 0.002
    slow_target_fraction: float = 1.0


@dataclass(frozen=True)
class DreamerTaskRewardConfig:
    map_scale_m: tuple[float, float]
    step_scale_m: float
    move_penalty: float
    outage_penalty: float
    high_outage_threshold: float
    high_outage_reward_penalty: float
    distance_shaping_weight: float
    terminal_reward: float
    goal_tolerance_m: float


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


@dataclass
class DreamerLosses:
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    metrics: dict[str, float]


def build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ELU())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class DiscreteDreamerActor(nn.Module):
    """Categorical actor over the UAV discrete action space."""

    def __init__(self, feat_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (400, 400, 400, 400)) -> None:
        super().__init__()
        self.net = build_mlp(feat_dim, action_dim, hidden_dims)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())

    def distribution(self, features: torch.Tensor) -> Categorical:
        return Categorical(logits=self(features))

    def act(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.distribution(features)
        if deterministic:
            return torch.argmax(dist.logits, dim=-1)
        return dist.sample()


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


def detach_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach() for key, value in state.items()}


def flatten_state_sequence(
    state: dict[str, torch.Tensor],
    start: int = 0,
    end: int | None = None,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Flatten [B, T, ...] RSSM state sequences into [N, ...] start states."""

    sliced = {key: value[:, start:end] for key, value in state.items()}
    flat = {key: value.reshape(-1, value.shape[-1]) for key, value in sliced.items()}
    if mask is None:
        return flat
    flat_mask = mask[:, start:end].reshape(-1).bool()
    return {key: value[flat_mask] for key, value in flat.items()}


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


def navigation_rewards_from_predictions(
    observations: torch.Tensor,
    outages: torch.Tensor,
    config: DreamerTaskRewardConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute task rewards from predicted observations and outages.

    Args:
        observations: [H + 1, B, obs_dim], including the start observation.
        outages: [H, B], outage predicted for next observations.
    Returns:
        rewards: [H, B]
        terminals: [H, B]
    """

    if observations.shape[0] != outages.shape[0] + 1:
        raise ValueError(
            "observations must contain one more time step than outages; "
            f"got {observations.shape[0]} and {outages.shape[0]}"
        )
    scale = observations.new_tensor(config.map_scale_m, dtype=observations.dtype)
    goal_delta_m = observations[..., 2:4] * scale
    distances = torch.linalg.norm(goal_delta_m, dim=-1)
    progress = distances[:-1] - distances[1:]
    next_distances = distances[1:]
    clipped_outages = torch.clamp(outages, 0.0, 1.0)
    rewards = -float(config.move_penalty) - float(config.outage_penalty) * clipped_outages
    high_outage_excess = torch.relu(clipped_outages - float(config.high_outage_threshold))
    rewards = rewards - float(config.high_outage_reward_penalty) * high_outage_excess
    rewards = rewards + float(config.distance_shaping_weight) * progress / max(float(config.step_scale_m), 1.0)
    terminals = next_distances <= float(config.goal_tolerance_m)
    rewards = rewards + terminals.to(rewards.dtype) * float(config.terminal_reward)
    return rewards, terminals


def cumulative_discount_weights(discounts: torch.Tensor) -> torch.Tensor:
    """Return Dreamer-style loss weights from previous continuation discounts."""

    ones = torch.ones_like(discounts[:1])
    if discounts.shape[0] == 1:
        return ones
    return torch.cumprod(torch.cat([ones, discounts[:-1]], dim=0), dim=0)


def imagine_trajectory(
    world_model: RSSMWorldModel,
    actor: DiscreteDreamerActor,
    start_state: dict[str, torch.Tensor],
    horizon: int,
    reward_config: DreamerTaskRewardConfig,
    discount: float = 0.99,
    sample_state: bool = False,
) -> DreamerImagination:
    """Roll out a Dreamer-style latent trajectory from posterior start states.

    The RSSM can be frozen by the caller. This function keeps gradients only
    through actor log-probabilities and critic calls made later; RSSM states and
    predictions are treated as imagined data.
    """

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    current = detach_state(start_state)
    features = [world_model.rssm.get_feat(current).detach()]
    with torch.no_grad():
        start_pred = world_model.heads_from_feat(features[-1])
    observations = [start_pred["observations"].detach()]
    actions = []
    log_probs = []
    entropies = []
    outages = []
    done_probs = []

    for _ in range(int(horizon)):
        feat = features[-1]
        dist = actor.distribution(feat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        with torch.no_grad():
            current = world_model.rssm.img_step(current, action, sample=sample_state)
            next_feat = world_model.rssm.get_feat(current)
            pred = world_model.heads_from_feat(next_feat)
            next_observation = pred["observations"]
            outage = pred["outages"]
            done_prob = torch.sigmoid(pred["done_logits"])
        features.append(next_feat.detach())
        observations.append(next_observation.detach())
        actions.append(action)
        log_probs.append(log_prob)
        entropies.append(entropy)
        outages.append(outage.detach())
        done_probs.append(done_prob.detach())

    feature_tensor = torch.stack(features, dim=0)
    observation_tensor = torch.stack(observations, dim=0)
    outage_tensor = torch.stack(outages, dim=0)
    done_prob_tensor = torch.stack(done_probs, dim=0)
    rewards, terminals = navigation_rewards_from_predictions(observation_tensor, outage_tensor, reward_config)
    continuation = 1.0 - torch.maximum(done_prob_tensor, terminals.to(done_prob_tensor.dtype))
    discounts = float(discount) * continuation
    weights = cumulative_discount_weights(discounts).detach()
    return DreamerImagination(
        features=feature_tensor,
        actions=torch.stack(actions, dim=0),
        log_probs=torch.stack(log_probs, dim=0),
        entropies=torch.stack(entropies, dim=0),
        pred_observations=observation_tensor,
        pred_outages=outage_tensor,
        done_probs=done_prob_tensor,
        rewards=rewards.detach(),
        discounts=discounts.detach(),
        weights=weights,
    )


def dreamer_actor_critic_losses(
    imagination: DreamerImagination,
    critic: DreamerCritic,
    target_critic: DreamerCritic,
    config: DreamerActorCriticConfig,
    nav_prior_weight: float = 0.0,
) -> DreamerLosses:
    """Compute discrete DreamerV2-style actor and critic losses."""

    features = imagination.features
    rewards = imagination.rewards
    discounts = imagination.discounts
    weights = imagination.weights.detach()
    values = critic(features[:-1])
    with torch.no_grad():
        target_values = target_critic(features[:-1])
        bootstrap = target_critic(features[-1])
        returns = lambda_return(
            rewards=rewards,
            values=target_values,
            discounts=discounts,
            bootstrap=bootstrap,
            lambda_=config.lambda_,
        )
        advantages = returns - target_values

    actor_objective = imagination.log_probs * advantages + float(config.actor_entropy) * imagination.entropies
    actor_base_loss = -(weights * actor_objective).mean()
    actor_loss = actor_base_loss
    nav_prior_loss = rewards.new_tensor(0.0)
    nav_prior_match_rate = 0.0
    if (
        float(nav_prior_weight) > 0.0
        and imagination.actor_probs is not None
        and imagination.nav_prior_probs is not None
    ):
        actor_probs = imagination.actor_probs.clamp_min(1.0e-8)
        prior_probs = imagination.nav_prior_probs.detach()
        nav_cross_entropy = -(prior_probs * torch.log(actor_probs)).sum(dim=-1)
        nav_prior_loss = (weights * nav_cross_entropy).mean()
        actor_loss = actor_loss + float(nav_prior_weight) * nav_prior_loss
        actor_best = torch.argmax(actor_probs.detach(), dim=-1)
        prior_best = torch.argmax(prior_probs, dim=-1)
        nav_prior_match_rate = float((actor_best == prior_best).float().mean().detach().cpu())
    critic_loss = (weights * F.mse_loss(values, returns.detach(), reduction="none")).mean()
    metrics = {
        "actor_loss": float(actor_loss.detach().cpu()),
        "actor_base_loss": float(actor_base_loss.detach().cpu()),
        "actor_nav_prior_loss": float(nav_prior_loss.detach().cpu()),
        "actor_nav_prior_weight": float(nav_prior_weight),
        "actor_nav_prior_match_rate": float(nav_prior_match_rate),
        "critic_loss": float(critic_loss.detach().cpu()),
        "imagined_return_mean": float(returns.detach().mean().cpu()),
        "imagined_reward_mean": float(rewards.detach().mean().cpu()),
        "imagined_discount_mean": float(discounts.detach().mean().cpu()),
        "actor_entropy": float(imagination.entropies.detach().mean().cpu()),
        "critic_value_mean": float(values.detach().mean().cpu()),
    }
    return DreamerLosses(
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        returns=returns,
        values=values,
        metrics=metrics,
    )


def update_slow_target(source: nn.Module, target: nn.Module, fraction: float) -> None:
    """Dreamer-style slow target update.

    fraction=1.0 copies source into target. Smaller values perform Polyak
    averaging: target <- fraction * source + (1 - fraction) * target.
    """

    mix = float(fraction)
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - mix).add_(source_param.data, alpha=mix)
