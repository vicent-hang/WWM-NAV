
from __future__ import annotations

from argparse import Namespace

import torch
from torch.nn import functional as F

from wwm_nav.models.rssm import RSSMWorldModel, rssm_kl
from wwm_nav.radiomap import RadioMap


def reward_normalization_stats(rewards, args: Namespace) -> tuple[float, float]:
    if not getattr(args, "reward_normalize", False):
        return 0.0, 1.0
    mean = float(rewards.mean())
    std = float(rewards.std())
    return mean, max(std, 1.0e-6)


def transform_rewards(rewards, args: Namespace, mean: float, std: float):
    result = rewards
    if getattr(args, "reward_clip", False):
        result = torch.clamp(result, float(args.reward_clip_min), float(args.reward_clip_max))
    if getattr(args, "reward_normalize", False):
        result = (result - float(mean)) / max(float(std), 1.0e-6)
    return result


def inverse_transform_rewards(rewards, mean: float, std: float, normalize: bool):
    if normalize:
        return rewards * max(float(std), 1.0e-6) + float(mean)
    return rewards


def next_xy_error_m(pred_observations, true_observations, radio_map: RadioMap):
    scale = pred_observations.new_tensor(
        [radio_map.x_m[-1] - radio_map.x_m[0], radio_map.y_m[-1] - radio_map.y_m[0]]
    ).clamp_min(1.0)
    pred_xy = pred_observations[..., 0:2] * scale
    true_xy = true_observations[..., 0:2] * scale
    return torch.linalg.norm(pred_xy - true_xy, dim=-1).mean()


def rssm_loss(
    model: RSSMWorldModel,
    batch: dict[str, object],
    args: Namespace,
    reward_mean: float,
    reward_std: float,
    radio_map: RadioMap,
    sample: bool,
    collect_metrics: bool = True,
):
    observations = batch["observations"]
    actions = batch["actions"]
    physical_contexts = batch.get("physical_contexts")
    building_patches = batch.get("building_patches")
    rewards = transform_rewards(batch["rewards"], args, reward_mean, reward_std)
    outages = batch["outages"]
    dones = batch["dones"]
    targets = observations[:, 1:]

    post, prior, post_pred, prior_pred = model.observe_sequence(
        observations,
        actions,
        physical_contexts=physical_contexts,
        building_patches=building_patches,
        sample=sample,
    )
    high_targets = (outages >= float(args.high_outage_threshold)).float()

    post_obs_loss = F.mse_loss(post_pred["observations"], targets)
    post_reward_loss = F.mse_loss(post_pred["rewards"], rewards)
    post_outage_loss = F.mse_loss(post_pred["outages"], outages)
    post_high_loss = F.binary_cross_entropy_with_logits(post_pred["high_outage_logits"], high_targets)
    post_done_loss = F.binary_cross_entropy_with_logits(post_pred["done_logits"], dones)

    prior_obs_loss = F.mse_loss(prior_pred["observations"], targets)
    prior_reward_loss = F.mse_loss(prior_pred["rewards"], rewards)
    prior_outage_loss = F.mse_loss(prior_pred["outages"], outages)
    prior_high_loss = F.binary_cross_entropy_with_logits(prior_pred["high_outage_logits"], high_targets)
    prior_done_loss = F.binary_cross_entropy_with_logits(prior_pred["done_logits"], dones)

    kl_value = rssm_kl(model.rssm, post, prior)
    raw_kl_loss = kl_value.mean()
    kl_loss = kl_value.clamp(min=float(args.free_nats)).mean()
    post_sg = {key: value.detach() for key, value in post.items()}
    prior_sg = {key: value.detach() for key, value in prior.items()}
    dynamics_kl_loss = rssm_kl(model.rssm, post_sg, prior).clamp(min=float(args.free_nats)).mean()
    representation_kl_loss = rssm_kl(model.rssm, post, prior_sg).clamp(min=float(args.free_nats)).mean()
    if getattr(args, "kl_balance_style", "dreamerv3") == "dreamerv3":
        kl_term = float(args.dynamics_kl_weight) * dynamics_kl_loss + float(args.representation_kl_weight) * representation_kl_loss
    else:
        kl_term = float(args.kl_weight) * kl_loss

    posterior_loss = (
        float(args.obs_weight) * post_obs_loss
        + float(args.reward_weight) * post_reward_loss
        + float(args.outage_weight) * post_outage_loss
        + float(args.high_outage_head_weight) * post_high_loss
        + float(args.done_weight) * post_done_loss
    )
    prior_prediction_loss = (
        float(args.prior_obs_weight) * prior_obs_loss
        + float(args.prior_reward_weight) * prior_reward_loss
        + float(args.prior_outage_weight) * prior_outage_loss
        + float(args.high_outage_head_weight) * prior_high_loss
        + float(args.prior_done_weight) * prior_done_loss
    )
    loss = posterior_loss + prior_prediction_loss + kl_term
    if not collect_metrics:
        return loss, {}

    post_rewards = inverse_transform_rewards(post_pred["rewards"], reward_mean, reward_std, bool(args.reward_normalize))
    prior_rewards = inverse_transform_rewards(prior_pred["rewards"], reward_mean, reward_std, bool(args.reward_normalize))
    post_high_probs = torch.sigmoid(post_pred["high_outage_logits"])
    prior_high_probs = torch.sigmoid(prior_pred["high_outage_logits"])
    metrics = {
        "loss": float(loss.detach().cpu()),
        "posterior_loss": float(posterior_loss.detach().cpu()),
        "prior_prediction_loss": float(prior_prediction_loss.detach().cpu()),
        "post_obs_loss": float(post_obs_loss.detach().cpu()),
        "post_reward_loss": float(post_reward_loss.detach().cpu()),
        "post_outage_loss": float(post_outage_loss.detach().cpu()),
        "post_high_outage_loss": float(post_high_loss.detach().cpu()),
        "post_done_loss": float(post_done_loss.detach().cpu()),
        "prior_obs_loss": float(prior_obs_loss.detach().cpu()),
        "prior_reward_loss": float(prior_reward_loss.detach().cpu()),
        "prior_outage_loss": float(prior_outage_loss.detach().cpu()),
        "prior_high_outage_loss": float(prior_high_loss.detach().cpu()),
        "prior_done_loss": float(prior_done_loss.detach().cpu()),
        "raw_kl_loss": float(raw_kl_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "post_reward_mae": float((post_rewards - batch["rewards"]).abs().mean().detach().cpu()),
        "post_outage_mae": float((post_pred["outages"] - outages).abs().mean().detach().cpu()),
        "post_high_outage_prob_mae": float((post_high_probs - high_targets).abs().mean().detach().cpu()),
        "post_xy_error_m": float(next_xy_error_m(post_pred["observations"], targets, radio_map).detach().cpu()),
        "prior_reward_mae": float((prior_rewards - batch["rewards"]).abs().mean().detach().cpu()),
        "prior_outage_mae": float((prior_pred["outages"] - outages).abs().mean().detach().cpu()),
        "prior_high_outage_prob_mae": float((prior_high_probs - high_targets).abs().mean().detach().cpu()),
        "prior_xy_error_m": float(next_xy_error_m(prior_pred["observations"], targets, radio_map).detach().cpu()),
    }
    return loss, metrics
