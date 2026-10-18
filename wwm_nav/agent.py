
from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from wwm_nav.config import NavigationConfig
from wwm_nav.env import ACTIONS, GridNavigationEnv
from wwm_nav.models.dreamer_actor_critic import (
    DreamerCritic,
    DreamerImagination,
    DreamerV3DiscreteActor,
    EMAScalarNormalizer,
    lambda_return,
    update_slow_target,
)
from wwm_nav.models.losses import rssm_loss
from wwm_nav.models.q_network import DuelingQNetwork
from wwm_nav.models.rssm import RSSMConfig, RSSMWorldModel
from wwm_nav.radiomap import RadioMap
from wwm_nav.tools.plotting import (
    plot_wwm_nav_actor_critic_training,
    plot_wwm_nav_training_history,
    plot_wwm_nav_world_model_training,
)
from wwm_nav.wwm_nav_map import (
    file_sha256,
    semantic_npz_sha256,
    validate_wwm_nav_generated_map,
)

PREVIOUS_ACTION_OFFSET = 6
METHOD_NAME = "WWM-NAV"
EXPERIMENT_CONTRACT = "wwm_nav_train_then_final_eval_v1"
LATENT_REFRESH_INTERVAL = 25


@dataclass(frozen=True)
class WWMNavConfig:
    radiomap_path: Path
    save_dir: Path = Path("outputs/wwm_nav_seed1")
    start_m: tuple[float, float] = (100.0, 100.0)
    goal_m: tuple[float, float] = (1400.0, 1600.0)
    train_goal_mode: str = "random_starts_fixed_goal"
    min_initial_distance_m: float = 800.0
    seed: int = 1
    episodes: int = 1500
    max_steps: int = 300
    replay_size: int = 100_000
    seq_len: int = 32
    batch_size: int = 128
    actor_batch_size: int = 128
    rssm_updates_per_step: int = 1
    actor_updates_per_step: int = 1
    outage_penalty: float = 5.0
    high_outage_reward_penalty: float = 0.0
    high_outage_threshold: float = 0.45
    terminal_reward: float = 200.0
    distance_shaping_weight: float = 2.0
    gamma: float = 0.99
    rssm_lr: float = 1.0e-4
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    rssm_embed_dim: int = 64
    rssm_deter_dim: int = 128
    rssm_stoch_dim: int = 32
    rssm_hidden_dim: int = 128
    rssm_classes: int = 16
    rssm_unimix: float = 0.01
    physical_num_bs: int = 10
    physical_context_embed_dim: int = 64
    building_patch_size: int = 9
    building_patch_embed_dim: int = 64
    obs_weight: float = 1.0
    reward_weight: float = 0.05
    outage_weight: float = 5.0
    high_outage_head_weight: float = 2.0
    done_weight: float = 0.1
    prior_obs_weight: float = 1.0
    prior_reward_weight: float = 0.05
    prior_outage_weight: float = 5.0
    prior_done_weight: float = 0.1
    kl_balance_style: str = "dreamerv3"
    kl_weight: float = 1.0
    dynamics_kl_weight: float = 0.5
    representation_kl_weight: float = 0.1
    free_nats: float = 1.0
    reward_clip: bool = True
    reward_clip_min: float = -200.0
    reward_clip_max: float = 200.0
    reward_normalize: bool = False
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    q_hidden_dims: tuple[int, ...] = (256, 256)
    actor_imagine_horizon: int = 10
    actor_entropy_weight: float = 0.002
    actor_lambda: float = 0.95
    actor_target_update_fraction: float = 0.01
    actor_grad_clip: float = 100.0
    actor_unimix: float = 0.01
    actor_outscale: float = 0.01
    actor_return_norm_rate: float = 0.01
    actor_advantage_norm_rate: float = 0.01
    actor_norm_min_scale: float = 1.0e-4
    actor_goal_progress_weight: float = 0.0
    actor_goal_progress_clip: float = 1.0
    actor_goal_terminal_bonus: float = 0.0
    actor_navigation_prior: bool = True
    actor_navigation_prior_style: str = "snarm_q_hard"
    actor_navigation_prior_weight: float = 0.1
    actor_navigation_prior_min_weight: float = 0.1
    actor_navigation_prior_decay_steps: int = 0
    actor_navigation_prior_temperature_m: float = 80.0
    actor_navigation_prior_init_actor: bool = True
    actor_navigation_prior_pretrain_updates: int = 5_000
    actor_navigation_prior_pretrain_samples: int = 100_000
    actor_navigation_prior_pretrain_batch_size: int = 256
    actor_navigation_prior_init_critic: bool = True
    actor_navigation_prior_critic_updates: int = 5_000
    actor_navigation_prior_critic_samples: int = 100_000
    actor_navigation_prior_critic_batch_size: int = 256
    sample_rssm_state: bool = False
    mask_outbound_actions: bool = True
    checkpoint_every_episodes: int = 20
    strict_wwm_nav_contract: bool = True
    resume_checkpoint_path: Path | None = None
    device: str = "cuda"

    @property
    def observation_dim(self) -> int:
        return 2 + 2 + 1 + 1 + len(ACTIONS) + 1

    @property
    def action_dim(self) -> int:
        return len(ACTIONS)


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return np.random.default_rng(int(seed))


def serializable_config(config: WWMNavConfig) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def validate_wwm_nav_config(config: WWMNavConfig) -> None:
    if not config.strict_wwm_nav_contract:
        return
    expected = {
        "start_m": (100.0, 100.0),
        "goal_m": (1400.0, 1600.0),
        "train_goal_mode": "random_starts_fixed_goal",
        "min_initial_distance_m": 800.0,
        "seed": 1,
        "episodes": 1500,
        "max_steps": 300,
        "replay_size": 100_000,
        "seq_len": 32,
        "batch_size": 128,
        "actor_batch_size": 128,
        "rssm_updates_per_step": 1,
        "actor_updates_per_step": 1,
        "outage_penalty": 5.0,
        "high_outage_reward_penalty": 0.0,
        "high_outage_threshold": 0.45,
        "terminal_reward": 200.0,
        "distance_shaping_weight": 2.0,
        "gamma": 0.99,
        "rssm_lr": 1.0e-4,
        "actor_lr": 1.0e-4,
        "critic_lr": 1.0e-4,
        "rssm_deter_dim": 128,
        "rssm_stoch_dim": 32,
        "rssm_classes": 16,
        "rssm_hidden_dim": 128,
        "rssm_embed_dim": 64,
        "rssm_unimix": 0.01,
        "obs_weight": 1.0,
        "reward_weight": 0.05,
        "outage_weight": 5.0,
        "done_weight": 0.1,
        "prior_obs_weight": 1.0,
        "prior_reward_weight": 0.05,
        "prior_outage_weight": 5.0,
        "prior_done_weight": 0.1,
        "kl_balance_style": "dreamerv3",
        "kl_weight": 1.0,
        "dynamics_kl_weight": 0.5,
        "representation_kl_weight": 0.1,
        "free_nats": 1.0,
        "high_outage_head_weight": 2.0,
        "reward_clip": True,
        "reward_clip_min": -200.0,
        "reward_clip_max": 200.0,
        "reward_normalize": False,
        "actor_imagine_horizon": 10,
        "actor_hidden_dims": (256, 256),
        "q_hidden_dims": (256, 256),
        "actor_entropy_weight": 0.002,
        "actor_lambda": 0.95,
        "actor_target_update_fraction": 0.01,
        "actor_grad_clip": 100.0,
        "actor_unimix": 0.01,
        "actor_outscale": 0.01,
        "actor_return_norm_rate": 0.01,
        "actor_advantage_norm_rate": 0.01,
        "actor_norm_min_scale": 1.0e-4,
        "actor_goal_progress_weight": 0.0,
        "actor_goal_progress_clip": 1.0,
        "actor_goal_terminal_bonus": 0.0,
        "actor_navigation_prior": True,
        "actor_navigation_prior_style": "snarm_q_hard",
        "actor_navigation_prior_weight": 0.1,
        "actor_navigation_prior_min_weight": 0.1,
        "actor_navigation_prior_decay_steps": 0,
        "actor_navigation_prior_temperature_m": 80.0,
        "actor_navigation_prior_init_actor": True,
        "actor_navigation_prior_pretrain_updates": 5_000,
        "actor_navigation_prior_pretrain_samples": 100_000,
        "actor_navigation_prior_pretrain_batch_size": 256,
        "actor_navigation_prior_init_critic": True,
        "actor_navigation_prior_critic_updates": 5_000,
        "actor_navigation_prior_critic_samples": 100_000,
        "actor_navigation_prior_critic_batch_size": 256,
        "physical_num_bs": 10,
        "physical_context_embed_dim": 64,
        "building_patch_size": 9,
        "building_patch_embed_dim": 64,
        "sample_rssm_state": False,
        "mask_outbound_actions": True,
        "checkpoint_every_episodes": 20,
        "strict_wwm_nav_contract": True,
        "device": "cuda",
    }
    mismatches = {
        key: {"expected": expected_value, "actual": getattr(config, key)}
        for key, expected_value in expected.items()
        if getattr(config, key) != expected_value
    }
    if mismatches:
        raise ValueError(f"strict WWM-NAV configuration mismatch: {mismatches}")
    validate_wwm_nav_generated_map(config.radiomap_path)


class EpisodeSequenceReplay:
    def __init__(self, capacity_steps: int) -> None:
        self.capacity_steps = int(capacity_steps)
        self.episodes: list[dict[str, np.ndarray]] = []
        self.current: dict[str, list[Any]] | None = None
        self.total_steps = 0

    def start_episode(self, initial_obs: np.ndarray, state_xy_m: np.ndarray, goal_xy_m: np.ndarray) -> None:
        self.end_episode()
        self.current = {
            "observations": [np.asarray(initial_obs, dtype=np.float32).copy()],
            "actions": [],
            "rewards": [],
            "outages": [],
            "dones": [],
            "state_xy_m": [np.asarray(state_xy_m, dtype=np.float32).copy()],
            "goal_xy_m": np.asarray(goal_xy_m, dtype=np.float32).copy(),
        }

    def append(self, action: int, reward: float, outage: float, next_obs: np.ndarray, next_state_xy_m: np.ndarray, done: bool) -> None:
        if self.current is None:
            raise RuntimeError("start_episode must be called before append")
        self.current["actions"].append(int(action))
        self.current["rewards"].append(float(reward))
        self.current["outages"].append(float(outage))
        self.current["dones"].append(float(done))
        self.current["observations"].append(np.asarray(next_obs, dtype=np.float32).copy())
        self.current["state_xy_m"].append(np.asarray(next_state_xy_m, dtype=np.float32).copy())
        self.total_steps += 1
        if done:
            self.end_episode()

    def end_episode(self) -> None:
        if self.current is None:
            return
        if self.current["actions"]:
            self.episodes.append({
                "observations": np.asarray(self.current["observations"], dtype=np.float32),
                "actions": np.asarray(self.current["actions"], dtype=np.int64),
                "rewards": np.asarray(self.current["rewards"], dtype=np.float32),
                "outages": np.asarray(self.current["outages"], dtype=np.float32),
                "dones": np.asarray(self.current["dones"], dtype=np.float32),
                "state_xy_m": np.asarray(self.current["state_xy_m"], dtype=np.float32),
                "goal_xy_m": np.asarray(self.current["goal_xy_m"], dtype=np.float32),
            })
            while self.episodes and self.total_steps > self.capacity_steps:
                removed = self.episodes.pop(0)
                self.total_steps -= int(removed["actions"].shape[0])
        self.current = None

    def _sampleable_episodes(self, seq_len: int, include_current: bool = True) -> list[dict[str, np.ndarray]]:
        items = [ep for ep in self.episodes if ep["actions"].shape[0] >= seq_len]
        if include_current and self.current is not None and len(self.current["actions"]) >= seq_len:
            items.append({
                "observations": np.asarray(self.current["observations"], dtype=np.float32),
                "actions": np.asarray(self.current["actions"], dtype=np.int64),
                "rewards": np.asarray(self.current["rewards"], dtype=np.float32),
                "outages": np.asarray(self.current["outages"], dtype=np.float32),
                "dones": np.asarray(self.current["dones"], dtype=np.float32),
                "state_xy_m": np.asarray(self.current["state_xy_m"], dtype=np.float32),
                "goal_xy_m": np.asarray(self.current["goal_xy_m"], dtype=np.float32),
            })
        return items

    def can_sample(self, batch_size: int, seq_len: int) -> bool:
        return bool(self._sampleable_episodes(seq_len)) and self.total_steps >= max(batch_size, seq_len)

    def sample_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator, device) -> dict[str, object]:
        episodes = self._sampleable_episodes(seq_len)
        if not episodes:
            raise RuntimeError("not enough sequence replay data to sample")
        obs_batch = []
        action_batch = []
        reward_batch = []
        outage_batch = []
        done_batch = []
        state_xy_batch = []
        goal_xy_batch = []
        for _ in range(batch_size):
            ep = episodes[int(rng.integers(len(episodes)))]
            start = int(rng.integers(int(ep["actions"].shape[0] - seq_len) + 1))
            end = start + seq_len
            obs_batch.append(ep["observations"][start : end + 1])
            action_batch.append(ep["actions"][start:end])
            reward_batch.append(ep["rewards"][start:end])
            outage_batch.append(ep["outages"][start:end])
            done_batch.append(ep["dones"][start:end])
            state_xy_batch.append(ep["state_xy_m"][start : end + 1])
            goal_xy_batch.append(ep["goal_xy_m"])
        return {
            "observations": torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.stack(action_batch), dtype=torch.long, device=device),
            "rewards": torch.as_tensor(np.stack(reward_batch), dtype=torch.float32, device=device),
            "outages": torch.as_tensor(np.stack(outage_batch), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.stack(done_batch), dtype=torch.float32, device=device),
            "state_xy_m": torch.as_tensor(np.stack(state_xy_batch), dtype=torch.float32, device=device),
            "goal_xy_m": torch.as_tensor(np.stack(goal_xy_batch), dtype=torch.float32, device=device),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity_steps": self.capacity_steps,
            "episodes": self.episodes,
            "current": self.current,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["capacity_steps"]) != self.capacity_steps:
            raise ValueError("resume replay capacity does not match the current configuration")
        self.episodes = list(state["episodes"])
        self.current = state["current"]
        self.total_steps = int(state["total_steps"])


def build_online_observation(env: GridNavigationEnv, state_m: np.ndarray, previous_action: int | None, step_fraction: float, config: WWMNavConfig) -> np.ndarray:
    base = env.observe(state_m)
    distance_to_goal_norm = np.asarray([np.linalg.norm(base[2:4])], dtype=np.float32)
    previous_action_onehot = np.zeros(config.action_dim, dtype=np.float32)
    if previous_action is not None:
        previous_action_onehot[int(previous_action)] = 1.0
    return np.concatenate([
        base[:4],
        distance_to_goal_norm,
        base[4:5],
        previous_action_onehot,
        np.asarray([np.clip(step_fraction, 0.0, 1.0)], dtype=np.float32),
    ]).astype(np.float32)


class PhysicalContextCache:
    def __init__(self, radio_map: RadioMap, config: WWMNavConfig) -> None:
        self.radio_map = radio_map
        self.config = config
        self.context_cache: dict[tuple[int, int], np.ndarray] = {}
        self.patch_cache: dict[tuple[int, int], np.ndarray] = {}

    def arrays(self, states_xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states_xy_m, dtype=np.float64)
        flat = states.reshape(-1, 2)
        keys = [self.radio_map.nearest_index(xy) for xy in flat]
        for key in dict.fromkeys(keys):
            if key not in self.context_cache:
                xy = self.radio_map.index_to_xy(key)
                self.context_cache[key] = self.radio_map.physical_context_at_xy(xy, num_bs=self.config.physical_num_bs)
                self.patch_cache[key] = self.radio_map.building_patch_at_xy(xy, patch_size=self.config.building_patch_size)
        context_dim = RadioMap.physical_context_dim(self.config.physical_num_bs)
        contexts = np.asarray([self.context_cache[key] for key in keys], dtype=np.float32)
        patches = np.asarray([self.patch_cache[key] for key in keys], dtype=np.float32)
        return (
            contexts.reshape(*states.shape[:-1], context_dim),
            patches.reshape(*states.shape[:-1], 1, self.config.building_patch_size, self.config.building_patch_size),
        )


def physical_context_tensors(radio_map: RadioMap, states_xy_m, config: WWMNavConfig, device, cache: PhysicalContextCache | None = None):
    if torch.is_tensor(states_xy_m):
        states_np = states_xy_m.detach().cpu().numpy()
    else:
        states_np = np.asarray(states_xy_m, dtype=np.float64)
    if cache is None:
        cache = PhysicalContextCache(radio_map, config)
    contexts, patches = cache.arrays(states_np)
    return (
        torch.as_tensor(contexts, dtype=torch.float32, device=device),
        torch.as_tensor(patches, dtype=torch.float32, device=device),
    )


def add_physical_context_to_batch(batch: dict[str, object], radio_map: RadioMap, config: WWMNavConfig, device, cache: PhysicalContextCache):
    contexts, patches = physical_context_tensors(radio_map, batch["state_xy_m"][:, 1:], config, device, cache)
    batch["physical_contexts"] = contexts
    batch["building_patches"] = patches
    return batch


def valid_action_mask_for_xy(radio_map: RadioMap, xy_m, device):
    action_table = torch.as_tensor(np.asarray(ACTIONS, dtype=np.float32) * np.asarray(radio_map.resolution_m, dtype=np.float32)[None, :], device=device)
    next_xy = xy_m.float()[:, None, :] + action_table[None, :, :]
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    return (next_xy[..., 0] >= x_min) & (next_xy[..., 0] <= x_max) & (next_xy[..., 1] >= y_min) & (next_xy[..., 1] <= y_max)


def snarm_q_prior_for_xy(radio_map: RadioMap, xy_m, goal_m, device):
    action_table = torch.as_tensor(np.asarray(ACTIONS, dtype=np.float32) * np.asarray(radio_map.resolution_m, dtype=np.float32)[None, :], device=device)
    next_xy = xy_m.float()[:, None, :] + action_table[None, :, :]
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    next_xy = next_xy.clone()
    next_xy[..., 0] = torch.clamp(next_xy[..., 0], float(x_min), float(x_max))
    next_xy[..., 1] = torch.clamp(next_xy[..., 1], float(y_min), float(y_max))
    distances = torch.linalg.norm(next_xy - goal_m.float()[:, None, :], dim=-1)
    step_scale = max(float(np.mean(radio_map.resolution_m)), 1.0)
    return -float(NavigationConfig().move_penalty) * distances / step_scale


def navigation_prior_probs_for_xy(radio_map: RadioMap, xy_m, goal_m, valid_mask, config: WWMNavConfig, device):
    q_prior = snarm_q_prior_for_xy(radio_map, xy_m, goal_m, device)
    step_scale = max(float(np.mean(radio_map.resolution_m)), 1.0)
    distances = -q_prior * step_scale / max(float(NavigationConfig().move_penalty), 1.0e-8)
    safe_mask = torch.where(valid_mask.sum(dim=-1, keepdim=True) == 0, torch.ones_like(valid_mask, dtype=torch.bool), valid_mask)
    if config.actor_navigation_prior_style == "snarm_q_hard":
        masked_q = torch.where(safe_mask, q_prior, torch.full_like(q_prior, -1.0e9))
        return F.one_hot(torch.argmax(masked_q, dim=-1), num_classes=config.action_dim).to(q_prior.dtype)
    if config.actor_navigation_prior_style == "snarm_q_soft":
        temperature = max(float(config.actor_navigation_prior_temperature_m) / step_scale, 1.0e-6)
        logits = q_prior / temperature
    elif config.actor_navigation_prior_style == "distance_soft":
        temperature = max(float(config.actor_navigation_prior_temperature_m), 1.0e-6)
        logits = -distances / temperature
    else:
        raise ValueError(f"unknown actor_navigation_prior_style: {config.actor_navigation_prior_style}")
    logits = torch.where(safe_mask, logits, torch.full_like(logits, -1.0e9))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.where(safe_mask, probs, torch.zeros_like(probs))
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def actor_navigation_prior_weight(config: WWMNavConfig, global_step: int) -> float:
    if not config.actor_navigation_prior:
        return 0.0
    initial = max(float(config.actor_navigation_prior_weight), 0.0)
    minimum = max(float(config.actor_navigation_prior_min_weight), 0.0)
    decay_steps = max(int(config.actor_navigation_prior_decay_steps), 0)
    if decay_steps <= 0 or initial <= minimum:
        return initial
    progress = min(max(float(global_step) / float(decay_steps), 0.0), 1.0)
    return minimum + (initial - minimum) * (1.0 - progress)


def initialize_actor_snarm_style(
    rssm_model,
    actor,
    actor_optimizer,
    radio_map: RadioMap,
    config: WWMNavConfig,
    rng: np.random.Generator,
    device,
    cache: PhysicalContextCache,
) -> dict[str, float]:
    if (
        not (config.actor_navigation_prior_init_actor or config.actor_navigation_prior)
        or config.actor_navigation_prior_pretrain_updates <= 0
        or config.actor_navigation_prior_pretrain_samples <= 0
    ):
        return {
            "actor_nav_prior_pretrain_loss": 0.0,
            "actor_nav_prior_pretrain_match_rate": 0.0,
        }

    sample_count = int(config.actor_navigation_prior_pretrain_samples)
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    env = make_env(radio_map, np.asarray(config.goal_m, dtype=np.float64), config)
    states = np.column_stack(
        (
            rng.uniform(x_min, x_max, size=sample_count),
            rng.uniform(y_min, y_max, size=sample_count),
        )
    ).astype(np.float64)
    states = np.asarray([env.snap_to_map(state) for state in states], dtype=np.float32)
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(
        np.asarray(config.goal_m, dtype=np.float32).reshape(1, 2),
        dtype=torch.float32,
        device=device,
    )
    batch_size = min(
        max(int(config.actor_navigation_prior_pretrain_batch_size), 1), sample_count
    )
    losses: list[float] = []
    match_rates: list[float] = []
    actor.train()
    total_updates = int(config.actor_navigation_prior_pretrain_updates)
    for update_index in range(total_updates):
        indices = torch.randint(0, sample_count, (batch_size,), device=device)
        xy_m = states_t[indices]
        goal_m = goal_t.expand(batch_size, -1)
        previous_action = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        step_fraction = torch.zeros(batch_size, dtype=torch.float32, device=device)
        latent_state = rssm_model.rssm.initial(batch_size, device)
        actor_features, _ = dreamer_actor_features(
            rssm_model,
            radio_map,
            latent_state,
            xy_m,
            goal_m,
            previous_action,
            step_fraction,
            device,
            config,
            cache,
        )
        valid_mask = (
            valid_action_mask_for_xy(radio_map, xy_m, device)
            if config.mask_outbound_actions
            else torch.ones(batch_size, config.action_dim, dtype=torch.bool, device=device)
        )
        prior_probs = navigation_prior_probs_for_xy(
            radio_map, xy_m, goal_m, valid_mask, config, device
        ).detach()
        distribution = actor.distribution(actor_features, valid_mask=valid_mask)
        actor_probs = distribution.probs.clamp_min(1.0e-8)
        if config.actor_navigation_prior_style == "snarm_q_hard":
            target = torch.argmax(prior_probs, dim=-1)
            loss = F.nll_loss(torch.log(actor_probs), target)
        else:
            loss = -(prior_probs * torch.log(actor_probs)).sum(dim=-1).mean()
        actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.actor_grad_clip)
        actor_optimizer.step()
        with torch.no_grad():
            updated = actor.distribution(actor_features, valid_mask=valid_mask)
            match_rate = (
                torch.argmax(updated.probs, dim=-1) == torch.argmax(prior_probs, dim=-1)
            ).float().mean()
        losses.append(float(loss.detach().cpu()))
        match_rates.append(float(match_rate.detach().cpu()))
        if (update_index + 1) % 500 == 0 or update_index + 1 == total_updates:
            print(
                "WWM-NAV actor-prior initialization: "
                f"{update_index + 1}/{total_updates}",
                flush=True,
            )
    return {
        "actor_nav_prior_pretrain_loss": float(np.mean(losses)) if losses else 0.0,
        "actor_nav_prior_pretrain_match_rate": (
            float(np.mean(match_rates)) if match_rates else 0.0
        ),
    }


def initialize_critic_snarm_style(
    rssm_model,
    critic,
    critic_optimizer,
    radio_map: RadioMap,
    config: WWMNavConfig,
    rng: np.random.Generator,
    device,
    cache: PhysicalContextCache,
) -> dict[str, float]:
    if (
        not config.actor_navigation_prior_init_critic
        or config.actor_navigation_prior_critic_updates <= 0
        or config.actor_navigation_prior_critic_samples <= 0
    ):
        return {
            "critic_nav_prior_init_loss": 0.0,
            "critic_nav_prior_init_mae": 0.0,
            "critic_nav_prior_init_target_mean": 0.0,
        }

    sample_count = int(config.actor_navigation_prior_critic_samples)
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    env = make_env(radio_map, np.asarray(config.goal_m, dtype=np.float64), config)
    states = np.column_stack(
        (
            rng.uniform(x_min, x_max, size=sample_count),
            rng.uniform(y_min, y_max, size=sample_count),
        )
    ).astype(np.float64)
    states = np.asarray([env.snap_to_map(state) for state in states], dtype=np.float32)
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(
        np.asarray(config.goal_m, dtype=np.float32).reshape(1, 2),
        dtype=torch.float32,
        device=device,
    )
    batch_size = min(
        max(int(config.actor_navigation_prior_critic_batch_size), 1), sample_count
    )
    losses: list[float] = []
    maes: list[float] = []
    target_means: list[float] = []
    critic.train()
    total_updates = int(config.actor_navigation_prior_critic_updates)
    for update_index in range(total_updates):
        indices = torch.randint(0, sample_count, (batch_size,), device=device)
        xy_m = states_t[indices]
        goal_m = goal_t.expand(batch_size, -1)
        previous_action = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        step_fraction = torch.zeros(batch_size, dtype=torch.float32, device=device)
        latent_state = rssm_model.rssm.initial(batch_size, device)
        critic_features, _ = dreamer_actor_features(
            rssm_model,
            radio_map,
            latent_state,
            xy_m,
            goal_m,
            previous_action,
            step_fraction,
            device,
            config,
            cache,
        )
        valid_mask = (
            valid_action_mask_for_xy(radio_map, xy_m, device)
            if config.mask_outbound_actions
            else torch.ones(batch_size, config.action_dim, dtype=torch.bool, device=device)
        )
        q_prior = snarm_q_prior_for_xy(radio_map, xy_m, goal_m, device)
        safe_mask = torch.where(
            valid_mask.sum(dim=-1, keepdim=True) == 0,
            torch.ones_like(valid_mask, dtype=torch.bool),
            valid_mask,
        )
        target = torch.where(
            safe_mask, q_prior, torch.full_like(q_prior, -1.0e9)
        ).max(dim=-1).values.detach()
        values = critic(critic_features)
        loss = F.mse_loss(values, target)
        critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.actor_grad_clip)
        critic_optimizer.step()
        with torch.no_grad():
            mae = torch.mean(torch.abs(critic(critic_features) - target))
        losses.append(float(loss.detach().cpu()))
        maes.append(float(mae.detach().cpu()))
        target_means.append(float(target.detach().mean().cpu()))
        if (update_index + 1) % 500 == 0 or update_index + 1 == total_updates:
            print(
                "WWM-NAV critic-prior initialization: "
                f"{update_index + 1}/{total_updates}",
                flush=True,
            )
    return {
        "critic_nav_prior_init_loss": float(np.mean(losses)) if losses else 0.0,
        "critic_nav_prior_init_mae": float(np.mean(maes)) if maes else 0.0,
        "critic_nav_prior_init_target_mean": (
            float(np.mean(target_means)) if target_means else 0.0
        ),
    }


def dreamer_navigation_features(radio_map: RadioMap, xy_m, goal_m, previous_actions, step_fraction, config: WWMNavConfig):
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    scale = xy_m.new_tensor([x_max - x_min, y_max - y_min]).clamp_min(1.0)
    origin = xy_m.new_tensor([x_min, y_min])
    xy_norm = (xy_m.float() - origin) / scale
    goal_delta_norm = (goal_m.float() - xy_m.float()) / scale
    distance_norm = torch.linalg.norm(goal_delta_norm, dim=-1, keepdim=True)
    safe_previous = previous_actions.long().clamp_min(0)
    previous_onehot = F.one_hot(safe_previous, num_classes=config.action_dim).float()
    previous_onehot = torch.where(previous_actions.long().unsqueeze(-1) >= 0, previous_onehot, torch.zeros_like(previous_onehot))
    return torch.cat([xy_norm, goal_delta_norm, distance_norm, previous_onehot, step_fraction.float().reshape(-1, 1).clamp(0.0, 1.0)], dim=-1)


def dreamer_actor_feature_dim(rssm_model: RSSMWorldModel, config: WWMNavConfig) -> int:
    nav_dim = 2 + 2 + 1 + config.action_dim + 1
    return int(rssm_model.feat_dim) + int(rssm_model.context_embed_dim) + nav_dim


def dreamer_actor_features(rssm_model: RSSMWorldModel, radio_map: RadioMap, state: dict[str, object], xy_m, goal_m, previous_actions, step_fraction, device, config: WWMNavConfig, cache: PhysicalContextCache):
    with torch.no_grad():
        physical_context, building_patch = physical_context_tensors(radio_map, xy_m, config, device, cache)
        context_embed = rssm_model.encode_physical_contexts(physical_context, building_patch)
        rssm_feat = rssm_model.rssm.get_feat(state)
    nav_feat = dreamer_navigation_features(radio_map, xy_m, goal_m, previous_actions, step_fraction, config)
    return torch.cat([rssm_feat.detach(), context_embed.detach(), nav_feat.detach()], dim=-1), context_embed


def make_env(radio_map: RadioMap, goal_m: np.ndarray, config: WWMNavConfig) -> GridNavigationEnv:
    nav_config = NavigationConfig(
        outage_penalty=config.outage_penalty,
        high_outage_threshold=config.high_outage_threshold,
        high_outage_reward_penalty=config.high_outage_reward_penalty,
        terminal_reward=config.terminal_reward,
        distance_shaping_weight=config.distance_shaping_weight,
        max_steps=config.max_steps,
    )
    return GridNavigationEnv(radio_map=radio_map, goal_m=goal_m, config=nav_config)


def sample_episode_task(radio_map: RadioMap, config: WWMNavConfig, rng: np.random.Generator):
    start = np.asarray(config.start_m, dtype=np.float64)
    goal = np.asarray(config.goal_m, dtype=np.float64)
    if config.train_goal_mode == "fixed":
        return start, goal
    if config.train_goal_mode != "random_starts_fixed_goal":
        raise ValueError("supported train_goal_mode values: fixed, random_starts_fixed_goal")
    for _ in range(10_000):
        start = sample_map_point(radio_map, rng, margin_m=50.0)
        if np.linalg.norm(goal - start) >= float(config.min_initial_distance_m):
            return start, goal
    raise RuntimeError("failed to sample a start far enough from the goal")


def sample_map_point(
    radio_map: RadioMap,
    rng: np.random.Generator,
    margin_m: float,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    valid_x = radio_map.x_m[
        (radio_map.x_m >= x_min + margin_m) & (radio_map.x_m <= x_max - margin_m)
    ]
    valid_y = radio_map.y_m[
        (radio_map.y_m >= y_min + margin_m) & (radio_map.y_m <= y_max - margin_m)
    ]
    return np.asarray([rng.choice(valid_x), rng.choice(valid_y)], dtype=np.float64)


def update_latent_with_feedback(rssm_model: RSSMWorldModel, latent_state: dict[str, object], radio_map: RadioMap, action: int, next_obs: np.ndarray, next_state_xy_m: np.ndarray, device, config: WWMNavConfig, cache: PhysicalContextCache):
    with torch.no_grad():
        obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        embed = rssm_model.encode_observations(obs_t)
        action_t = torch.as_tensor([action], dtype=torch.long, device=device)
        physical_context, building_patch = physical_context_tensors(radio_map, np.asarray(next_state_xy_m).reshape(1, 2), config, device, cache)
        context_embed = rssm_model.encode_physical_contexts(physical_context, building_patch)
        post, _ = rssm_model.rssm.obs_step(latent_state, action_t, embed, context=context_embed, sample=config.sample_rssm_state)
    return {key: value.detach() for key, value in post.items()}


def current_episode_latent(rssm_model: RSSMWorldModel, replay: EpisodeSequenceReplay, radio_map: RadioMap, device, config: WWMNavConfig, cache: PhysicalContextCache):
    if replay.current is None or not replay.current["actions"]:
        return rssm_model.rssm.initial(1, device), False
    observations = np.asarray(replay.current["observations"], dtype=np.float32)
    actions = np.asarray(replay.current["actions"], dtype=np.int64)
    states_xy = np.asarray(replay.current["state_xy_m"], dtype=np.float64)[1:]
    with torch.no_grad():
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
        physical_context, building_patch = physical_context_tensors(radio_map, states_xy[None, ...], config, device, cache)
        post = rssm_model.posterior_from_context(obs_t, act_t, physical_contexts=physical_context, building_patches=building_patch, sample=config.sample_rssm_state)
    return {key: value.detach() for key, value in post.items()}, True


def rssm_training_args(config: WWMNavConfig) -> Namespace:
    return Namespace(**{
        "obs_weight": config.obs_weight,
        "reward_weight": config.reward_weight,
        "outage_weight": config.outage_weight,
        "high_outage_head_weight": config.high_outage_head_weight,
        "done_weight": config.done_weight,
        "prior_obs_weight": config.prior_obs_weight,
        "prior_reward_weight": config.prior_reward_weight,
        "prior_outage_weight": config.prior_outage_weight,
        "prior_done_weight": config.prior_done_weight,
        "kl_balance_style": config.kl_balance_style,
        "kl_weight": config.kl_weight,
        "dynamics_kl_weight": config.dynamics_kl_weight,
        "representation_kl_weight": config.representation_kl_weight,
        "free_nats": config.free_nats,
        "high_outage_threshold": config.high_outage_threshold,
        "reward_clip": config.reward_clip,
        "reward_clip_min": config.reward_clip_min,
        "reward_clip_max": config.reward_clip_max,
        "reward_normalize": config.reward_normalize,
    })


def update_rssm_network(rssm_model, optimizer, replay, radio_map, rng, config, device, cache):
    if not replay.can_sample(config.batch_size, config.seq_len):
        return {}
    batch = replay.sample_batch(config.batch_size, config.seq_len, rng, device)
    batch = add_physical_context_to_batch(batch, radio_map, config, device, cache)
    args = rssm_training_args(config)
    loss, metrics = rssm_loss(rssm_model, batch, args, 0.0, 1.0, radio_map, sample=True)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rssm_model.parameters(), config.actor_grad_clip)
    optimizer.step()
    return metrics


def physics_dreamer_v3_imagine(rssm_model, actor, start_state, start_xy_m, goal_xy_m, previous_actions, step_fraction, radio_map, config, device, cache, torch_generator=None):
    horizon = int(config.actor_imagine_horizon)
    current = {key: value.detach() for key, value in start_state.items()}
    xy_m = start_xy_m.float()
    goal_m = goal_xy_m.float()
    prev_action = previous_actions.long()
    step_frac = step_fraction.float()
    features = []
    actions = []
    log_probs = []
    entropies = []
    actor_probs = []
    nav_prior_probs = []
    rewards = []
    outages = []
    done_probs = []
    reward_components = {
        "predicted_env_reward": [],
        "goal_progress_reward": [],
        "terminal_bonus": [],
        "total_reward": [],
    }
    for _ in range(horizon):
        actor_feat, _ = dreamer_actor_features(rssm_model, radio_map, current, xy_m, goal_m, prev_action, step_frac, device, config, cache)
        valid_mask = valid_action_mask_for_xy(radio_map, xy_m, device) if config.mask_outbound_actions else torch.ones(actor_feat.shape[0], config.action_dim, dtype=torch.bool, device=device)
        dist = actor.distribution(actor_feat, valid_mask=valid_mask)
        action = dist.sample() if torch_generator is None else torch.multinomial(dist.probs, num_samples=1, generator=torch_generator).squeeze(-1)
        action_probs = dist.probs
        prior_probs = navigation_prior_probs_for_xy(radio_map, xy_m, goal_m, valid_mask, config, device) if config.actor_navigation_prior else torch.zeros_like(action_probs)
        next_xy = deterministic_next_xy_tensor(radio_map, xy_m, action, device)
        current_distance = torch.linalg.norm(xy_m - goal_m, dim=-1)
        next_distance = torch.linalg.norm(next_xy - goal_m, dim=-1)
        next_step_frac = torch.clamp(step_frac + 1.0 / max(float(config.max_steps), 1.0), 0.0, 1.0)
        physical_context, building_patch = physical_context_tensors(radio_map, next_xy, config, device, cache)
        next_context = rssm_model.encode_physical_contexts(physical_context, building_patch)
        with torch.no_grad():
            current = rssm_model.rssm.img_step(current, action, context=next_context, sample=config.sample_rssm_state)
            pred = rssm_model.heads_from_feat(rssm_model.rssm.get_feat(current), next_context)
            predicted_env_reward = pred["rewards"].detach()
            outage = pred["outages"].detach().clamp(0.0, 1.0)
            done_prob = torch.sigmoid(pred["done_logits"]).detach()
            step_scale = max(float(np.mean(radio_map.resolution_m)), 1.0)
            progress = torch.clamp(
                (current_distance - next_distance) / step_scale,
                -float(config.actor_goal_progress_clip),
                float(config.actor_goal_progress_clip),
            )
            terminal = next_distance <= float(NavigationConfig().goal_tolerance_m)
            goal_progress_reward = float(config.actor_goal_progress_weight) * progress
            terminal_bonus = terminal.to(predicted_env_reward.dtype) * float(
                config.actor_goal_terminal_bonus
            )
            # The registered WWM-NAV DreamerV3 Actor path optimizes the RSSM
            # reward head directly; it has no additional high-outage penalty.
            reward = predicted_env_reward + goal_progress_reward + terminal_bonus
        features.append(actor_feat)
        actions.append(action)
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        actor_probs.append(action_probs)
        nav_prior_probs.append(prior_probs.detach())
        rewards.append(reward)
        outages.append(outage)
        done_probs.append(done_prob)
        reward_components["predicted_env_reward"].append(predicted_env_reward)
        reward_components["goal_progress_reward"].append(goal_progress_reward)
        reward_components["terminal_bonus"].append(terminal_bonus)
        reward_components["total_reward"].append(reward)
        xy_m = next_xy.detach()
        prev_action = action.detach()
        step_frac = next_step_frac
    final_feat, _ = dreamer_actor_features(rssm_model, radio_map, current, xy_m, goal_m, prev_action, step_frac, device, config, cache)
    done_tensor = torch.stack(done_probs, dim=0)
    discounts = float(config.gamma) * (1.0 - done_tensor)
    weights = torch.cumprod(torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]], dim=0), dim=0).detach()
    return DreamerImagination(
        features=torch.stack([*features, final_feat], dim=0),
        actions=torch.stack(actions, dim=0),
        log_probs=torch.stack(log_probs, dim=0),
        entropies=torch.stack(entropies, dim=0),
        pred_observations=torch.zeros(horizon + 1, start_xy_m.shape[0], config.observation_dim, dtype=start_xy_m.dtype, device=device),
        pred_outages=torch.stack(outages, dim=0),
        done_probs=done_tensor,
        rewards=torch.stack(rewards, dim=0),
        discounts=discounts.detach(),
        weights=weights,
        actor_probs=torch.stack(actor_probs, dim=0),
        nav_prior_probs=torch.stack(nav_prior_probs, dim=0),
        reward_components={key: torch.stack(value, dim=0) for key, value in reward_components.items()},
    )


def deterministic_next_xy_tensor(radio_map: RadioMap, xy_m, actions, device):
    action_table = torch.as_tensor(np.asarray(ACTIONS, dtype=np.float32) * np.asarray(radio_map.resolution_m, dtype=np.float32)[None, :], dtype=torch.float32, device=device)
    next_xy = xy_m.float() + action_table[actions.long()]
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    next_xy[:, 0] = torch.clamp(next_xy[:, 0], float(x_min), float(x_max))
    next_xy[:, 1] = torch.clamp(next_xy[:, 1], float(y_min), float(y_max))
    return next_xy


def dreamer_v3_actor_critic_losses(imagination, critic, target_critic, return_norm, advantage_norm, config, global_step):
    rewards = imagination.rewards
    discounts = imagination.discounts
    weights = imagination.weights.detach()
    values = critic(imagination.features[:-1])
    with torch.no_grad():
        target_values = target_critic(imagination.features[:-1])
        bootstrap = target_critic(imagination.features[-1])
        returns = lambda_return(rewards, target_values, discounts, bootstrap, config.actor_lambda)
        return_norm.update(returns)
        scaled_advantages = return_norm.normalize(returns) - return_norm.normalize(target_values)
        advantage_norm.update(scaled_advantages)
        actor_advantages = advantage_norm.normalize(scaled_advantages)
    actor_objective = imagination.log_probs * actor_advantages + float(config.actor_entropy_weight) * imagination.entropies
    actor_base_loss = -(weights * actor_objective).mean()
    actor_loss = actor_base_loss
    nav_prior_loss = rewards.new_tensor(0.0)
    nav_prior_weight = actor_navigation_prior_weight(config, global_step)
    nav_prior_match_rate = 0.0
    if nav_prior_weight > 0.0 and imagination.actor_probs is not None and imagination.nav_prior_probs is not None:
        actor_probs = imagination.actor_probs.clamp_min(1.0e-8)
        prior_probs = imagination.nav_prior_probs.detach()
        nav_cross_entropy = -(prior_probs * torch.log(actor_probs)).sum(dim=-1)
        nav_prior_loss = (weights * nav_cross_entropy).mean()
        actor_loss = actor_loss + float(nav_prior_weight) * nav_prior_loss
        nav_prior_match_rate = float((torch.argmax(actor_probs.detach(), dim=-1) == torch.argmax(prior_probs, dim=-1)).float().mean().detach().cpu())
    critic_loss = (weights * F.mse_loss(values, returns.detach(), reduction="none")).mean()
    return actor_loss, critic_loss, {
        "actor_loss": float(actor_loss.detach().cpu()),
        "actor_base_loss": float(actor_base_loss.detach().cpu()),
        "actor_nav_prior_loss": float(nav_prior_loss.detach().cpu()),
        "actor_nav_prior_weight": float(nav_prior_weight),
        "actor_nav_prior_match_rate": float(nav_prior_match_rate),
        "critic_loss": float(critic_loss.detach().cpu()),
        "imagined_return_mean": float(returns.detach().mean().cpu()),
        "imagined_reward_mean": float(rewards.detach().mean().cpu()),
        "actor_entropy": float(imagination.entropies.detach().mean().cpu()),
    }


def update_dreamer_v3_actor_critic(rssm_model, actor, critic, target_critic, return_norm, advantage_norm, actor_optimizer, critic_optimizer, replay, radio_map, rng, config, device, global_step, cache):
    if not replay.can_sample(config.actor_batch_size, config.seq_len):
        return {}
    batch = replay.sample_batch(config.actor_batch_size, config.seq_len, rng, device)
    batch = add_physical_context_to_batch(batch, radio_map, config, device, cache)
    with torch.no_grad():
        post, _, _, _ = rssm_model.observe_sequence(batch["observations"], batch["actions"], physical_contexts=batch["physical_contexts"], building_patches=batch["building_patches"], sample=False)
    valid_mask = batch["dones"] < 0.5
    flat_state = {key: value.reshape(-1, *value.shape[2:])[valid_mask.reshape(-1)] for key, value in post.items()}
    if not flat_state or next(iter(flat_state.values())).shape[0] == 0:
        return {}
    flat_xy = batch["state_xy_m"][:, 1:].reshape(-1, 2)[valid_mask.reshape(-1)]
    flat_goal = batch["goal_xy_m"][:, None, :].expand(-1, config.seq_len, -1).reshape(-1, 2)[valid_mask.reshape(-1)]
    flat_previous_action = batch["actions"].reshape(-1)[valid_mask.reshape(-1)]
    flat_step_fraction = batch["observations"][:, 1:, -1].reshape(-1)[valid_mask.reshape(-1)]
    start_count = flat_xy.shape[0]
    if start_count > config.actor_batch_size:
        indices = torch.randperm(start_count, device=device)[: config.actor_batch_size]
        flat_state = {key: value[indices] for key, value in flat_state.items()}
        flat_xy = flat_xy[indices]
        flat_goal = flat_goal[indices]
        flat_previous_action = flat_previous_action[indices]
        flat_step_fraction = flat_step_fraction[indices]
    imagination = physics_dreamer_v3_imagine(rssm_model, actor, flat_state, flat_xy, flat_goal, flat_previous_action, flat_step_fraction, radio_map, config, device, cache)
    actor_loss, critic_loss, metrics = dreamer_v3_actor_critic_losses(imagination, critic, target_critic, return_norm, advantage_norm, config, global_step)
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), config.actor_grad_clip)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), config.actor_grad_clip)
    actor_optimizer.step()
    critic_optimizer.step()
    update_slow_target(critic, target_critic, config.actor_target_update_fraction)
    return metrics


def choose_actor_action(rssm_model, actor, env, obs, state_m, latent_state, device, config, cache, deterministic: bool):
    previous_action = int(np.argmax(obs[PREVIOUS_ACTION_OFFSET : PREVIOUS_ACTION_OFFSET + config.action_dim])) if np.sum(obs[PREVIOUS_ACTION_OFFSET : PREVIOUS_ACTION_OFFSET + config.action_dim]) > 0 else -1
    step_fraction = float(obs[-1])
    with torch.no_grad():
        xy_m = torch.as_tensor(np.asarray(state_m, dtype=np.float32).reshape(1, 2), dtype=torch.float32, device=device)
        goal_m = torch.as_tensor(env.goal_m.astype(np.float32).reshape(1, 2), dtype=torch.float32, device=device)
        prev_action = torch.as_tensor([previous_action], dtype=torch.long, device=device)
        step_t = torch.as_tensor([step_fraction], dtype=torch.float32, device=device)
        actor_feat, _ = dreamer_actor_features(rssm_model, env.radio_map, latent_state, xy_m, goal_m, prev_action, step_t, device, config, cache)
        valid_mask = valid_action_mask_for_xy(env.radio_map, xy_m, device) if config.mask_outbound_actions else torch.ones(1, config.action_dim, dtype=torch.bool, device=device)
        dist = actor.distribution(actor_feat, valid_mask=valid_mask)
        action_t = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
    return int(action_t.detach().cpu().item())


def initialize_models(radio_map: RadioMap, config: WWMNavConfig, device):
    rssm_config = RSSMConfig(
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        embed_dim=config.rssm_embed_dim,
        deter_dim=config.rssm_deter_dim,
        stoch_dim=config.rssm_stoch_dim,
        hidden_dim=config.rssm_hidden_dim,
        classes=config.rssm_classes,
        unimix=config.rssm_unimix,
        physical_context_dim=RadioMap.physical_context_dim(config.physical_num_bs),
        physical_context_embed_dim=config.physical_context_embed_dim,
        building_patch_size=config.building_patch_size,
        building_patch_embed_dim=config.building_patch_embed_dim,
    )
    rssm_model = RSSMWorldModel(rssm_config).to(device)
    # The registered WWM-NAV implementation constructs two legacy Q networks
    # before the Actor. They are inactive in DreamerV3 Actor mode, but their
    # initialization advances Torch RNG and therefore belongs to the seed contract.
    DuelingQNetwork(
        config.observation_dim, config.action_dim, tuple(config.q_hidden_dims)
    ).to(device)
    DuelingQNetwork(
        config.observation_dim, config.action_dim, tuple(config.q_hidden_dims)
    ).to(device)
    actor_input_dim = dreamer_actor_feature_dim(rssm_model, config)
    actor = DreamerV3DiscreteActor(actor_input_dim, config.action_dim, tuple(config.actor_hidden_dims), config.actor_unimix, config.actor_outscale).to(device)
    critic = DreamerCritic(actor_input_dim, tuple(config.actor_hidden_dims)).to(device)
    target_critic = DreamerCritic(actor_input_dim, tuple(config.actor_hidden_dims)).to(device)
    target_critic.load_state_dict(critic.state_dict())
    return rssm_model, actor, critic, target_critic


def source_code_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]

    def git_output(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain", "--untracked-files=all")
    return {
        "code_commit": commit,
        "code_worktree_dirty": None if status is None else bool(status),
    }


def runtime_provenance(device) -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "resolved_device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def rollout_summary(
    rollout: dict[str, object], goal_m: np.ndarray, high_outage_threshold: float
) -> dict[str, float]:
    states = np.asarray(rollout["states"], dtype=np.float64)
    actions = np.asarray(rollout["actions"], dtype=np.int64)
    outages = np.asarray(rollout["outages"], dtype=np.float64)
    terminals = np.asarray(rollout["terminals"], dtype=np.bool_)
    success = bool(terminals[-1]) if terminals.size else False
    high_count = int(np.count_nonzero(outages >= high_outage_threshold)) if outages.size else 0
    return {
        "success": float(success),
        "steps": float(len(actions)),
        "final_distance": float(np.linalg.norm(states[-1] - goal_m)),
        "mean_outage": float(np.mean(outages)) if outages.size else 0.0,
        "max_outage": float(np.max(outages)) if outages.size else 0.0,
        "high_outage_count": float(high_count),
        "high_outage_rate": float(high_count / len(outages)) if outages.size else 0.0,
    }


def evaluate_models(
    config: WWMNavConfig,
    radio_map: RadioMap,
    rssm_model,
    actor,
    device,
    cache: PhysicalContextCache,
    deterministic: bool = True,
) -> dict[str, object]:
    env = make_env(radio_map, np.asarray(config.goal_m, dtype=np.float64), config)
    state = env.reset(start_m=np.asarray(config.start_m, dtype=np.float64))
    obs = build_online_observation(env, state, None, 0.0, config)
    latent_state = rssm_model.rssm.initial(1, device)
    states = [state.copy()]
    observations = [obs.copy()]
    actions: list[int] = []
    rewards: list[float] = []
    outages: list[float] = []
    terminals: list[bool] = []
    done = False
    with torch.inference_mode():
        while not done:
            action = choose_actor_action(
                rssm_model,
                actor,
                env,
                obs,
                state,
                latent_state,
                device,
                config,
                cache,
                deterministic=deterministic,
            )
            result = env.step_action(action)
            next_obs = build_online_observation(
                env,
                result.state,
                action,
                env._step_count / max(config.max_steps, 1),
                config,
            )
            latent_state = update_latent_with_feedback(
                rssm_model,
                latent_state,
                radio_map,
                action,
                next_obs,
                result.state,
                device,
                config,
                cache,
            )
            state = result.state
            obs = next_obs
            states.append(state.copy())
            observations.append(obs.copy())
            actions.append(action)
            rewards.append(result.reward)
            outages.append(result.outage)
            terminals.append(result.terminal)
            done = result.done
    return {
        "states": np.asarray(states, dtype=np.float64),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "outages": np.asarray(outages, dtype=np.float64),
        "terminals": np.asarray(terminals, dtype=np.bool_),
        "success": bool(result.terminal),
        "steps": int(env._step_count),
    }


def _resume_config_payload(config: WWMNavConfig) -> dict[str, object]:
    payload = serializable_config(config)
    payload.pop("resume_checkpoint_path", None)
    return payload


def save_training_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config: WWMNavConfig) -> dict[str, list[float]]:
    validate_wwm_nav_config(config)
    started_at = time.monotonic()
    print(
        f"WWM-NAV training started: episodes={config.episodes} "
        f"max_steps={config.max_steps} device={config.device}",
        flush=True,
    )
    rng = set_seed(config.seed)
    device = resolve_device(config.device)
    radio_map = RadioMap.from_npz(config.radiomap_path, require_physics=True)
    cache = PhysicalContextCache(radio_map, config)
    rssm_model, actor, critic, target_critic = initialize_models(radio_map, config, device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    return_norm = EMAScalarNormalizer(
        config.actor_return_norm_rate, config.actor_norm_min_scale
    ).to(device)
    advantage_norm = EMAScalarNormalizer(
        config.actor_advantage_norm_rate, config.actor_norm_min_scale
    ).to(device)
    replay = EpisodeSequenceReplay(config.replay_size)
    rssm_optimizer = torch.optim.Adam(rssm_model.parameters(), lr=config.rssm_lr)
    history: dict[str, list[float]] = {
        key: []
        for key in [
            "episode_return",
            "episode_steps",
            "success",
            "mean_outage",
            "max_outage",
            "high_outage_count",
            "high_outage_rate",
            "wm_loss",
            "rssm_prior_outage_mae",
            "actor_loss",
            "critic_loss",
            "imagined_return_mean",
            "measured_cells",
        ]
    }
    training_tasks: list[tuple[int, float, float, float, float]] = []
    trajectory_rows: list[list[float]] = []
    episode_successes: list[float] = []
    actor_init_metrics = {
        "actor_nav_prior_pretrain_loss": 0.0,
        "actor_nav_prior_pretrain_match_rate": 0.0,
    }
    critic_init_metrics = {
        "critic_nav_prior_init_loss": 0.0,
        "critic_nav_prior_init_mae": 0.0,
        "critic_nav_prior_init_target_mean": 0.0,
    }
    global_step = 0
    start_episode = 1

    if config.resume_checkpoint_path is None:
        print("WWM-NAV initializing navigation prior actor...", flush=True)
        actor_init_metrics = initialize_actor_snarm_style(
            rssm_model,
            actor,
            actor_optimizer,
            radio_map,
            config,
            rng,
            device,
            cache,
        )
        print("WWM-NAV initializing navigation prior critic...", flush=True)
        critic_init_metrics = initialize_critic_snarm_style(
            rssm_model,
            critic,
            critic_optimizer,
            radio_map,
            config,
            rng,
            device,
            cache,
        )
        target_critic.load_state_dict(critic.state_dict())
        print("WWM-NAV navigation prior initialization complete", flush=True)
    else:
        print(
            f"WWM-NAV resuming from {config.resume_checkpoint_path}",
            flush=True,
        )
        state = torch.load(
            config.resume_checkpoint_path, map_location=device, weights_only=False
        )
        if state.get("experiment_contract") != EXPERIMENT_CONTRACT:
            raise ValueError("resume checkpoint is not a compatible WWM-NAV state")
        if state.get("config") != _resume_config_payload(config):
            raise ValueError("resume checkpoint configuration does not match this run")
        if state.get("radiomap_semantic_sha256") != semantic_npz_sha256(
            config.radiomap_path
        ):
            raise ValueError("resume checkpoint radio-map identity mismatch")
        rssm_model.load_state_dict(state["rssm_model"])
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        target_critic.load_state_dict(state["target_critic"])
        rssm_optimizer.load_state_dict(state["rssm_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        return_norm.load_state_dict(state["return_norm"])
        advantage_norm.load_state_dict(state["advantage_norm"])
        replay.load_state_dict(state["replay"])
        history = state["history"]
        history.setdefault("measured_cells", [0.0] * len(history["episode_return"]))
        training_tasks = state["training_tasks"]
        trajectory_rows = state["trajectory_rows"]
        episode_successes = state["episode_successes"]
        actor_init_metrics = state["actor_init_metrics"]
        critic_init_metrics = state["critic_init_metrics"]
        global_step = int(state["global_step"])
        start_episode = int(state["episode"]) + 1
        rng.bit_generator.state = state["rng_state"]
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state["torch_cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])

    config.save_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(start_episode, config.episodes + 1):
        start_m, goal_m = sample_episode_task(radio_map, config, rng)
        training_tasks.append(
            (episode, float(start_m[0]), float(start_m[1]), float(goal_m[0]), float(goal_m[1]))
        )
        env = make_env(radio_map, goal_m, config)
        state_xy = env.reset(start_m=start_m)
        observation = build_online_observation(env, state_xy, None, 0.0, config)
        replay.start_episode(observation, state_xy, goal_m)
        latent_state = rssm_model.rssm.initial(1, device)
        episode_states = [state_xy.copy()]
        done = False
        episode_return = 0.0
        outages: list[float] = []
        rssm_losses: list[float] = []
        prior_outage_maes: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        imagined_returns: list[float] = []
        while not done:
            action = choose_actor_action(
                rssm_model,
                actor,
                env,
                observation,
                state_xy,
                latent_state,
                device,
                config,
                cache,
                deterministic=False,
            )
            result = env.step_action(action)
            next_observation = build_online_observation(
                env,
                result.state,
                action,
                env._step_count / max(config.max_steps, 1),
                config,
            )
            latent_state = update_latent_with_feedback(
                rssm_model,
                latent_state,
                radio_map,
                action,
                next_observation,
                result.state,
                device,
                config,
                cache,
            )
            replay.append(
                action,
                result.reward,
                result.outage,
                next_observation,
                result.state,
                result.done,
            )
            for _ in range(config.rssm_updates_per_step):
                metrics = update_rssm_network(
                    rssm_model,
                    rssm_optimizer,
                    replay,
                    radio_map,
                    rng,
                    config,
                    device,
                    cache,
                )
                if metrics:
                    rssm_losses.append(metrics["loss"])
                    prior_outage_maes.append(metrics["prior_outage_mae"])
            if global_step % LATENT_REFRESH_INTERVAL == 0:
                latent_state, _ = current_episode_latent(
                    rssm_model, replay, radio_map, device, config, cache
                )
            for _ in range(config.actor_updates_per_step):
                metrics = update_dreamer_v3_actor_critic(
                    rssm_model,
                    actor,
                    critic,
                    target_critic,
                    return_norm,
                    advantage_norm,
                    actor_optimizer,
                    critic_optimizer,
                    replay,
                    radio_map,
                    rng,
                    config,
                    device,
                    global_step,
                    cache,
                )
                if metrics:
                    actor_losses.append(metrics["actor_loss"])
                    critic_losses.append(metrics["critic_loss"])
                    imagined_returns.append(metrics["imagined_return_mean"])
            state_xy = result.state
            episode_states.append(state_xy.copy())
            observation = next_observation
            episode_return += result.reward
            outages.append(result.outage)
            done = result.done
            global_step += 1

        outage_array = np.asarray(outages, dtype=np.float64)
        history["episode_return"].append(float(episode_return))
        history["episode_steps"].append(float(env._step_count))
        history["success"].append(float(result.terminal))
        history["mean_outage"].append(
            float(np.mean(outage_array)) if outage_array.size else 0.0
        )
        history["max_outage"].append(
            float(np.max(outage_array)) if outage_array.size else 0.0
        )
        history["high_outage_count"].append(
            float(np.count_nonzero(outage_array >= config.high_outage_threshold))
        )
        history["high_outage_rate"].append(
            float(history["high_outage_count"][-1] / len(outage_array))
            if outage_array.size
            else 0.0
        )
        history["wm_loss"].append(
            float(np.mean(rssm_losses)) if rssm_losses else float("nan")
        )
        history["rssm_prior_outage_mae"].append(
            float(np.mean(prior_outage_maes)) if prior_outage_maes else float("nan")
        )
        history["actor_loss"].append(
            float(np.mean(actor_losses)) if actor_losses else 0.0
        )
        history["critic_loss"].append(
            float(np.mean(critic_losses)) if critic_losses else 0.0
        )
        history["imagined_return_mean"].append(
            float(np.mean(imagined_returns)) if imagined_returns else 0.0
        )
        history["measured_cells"].append(
            float(len({(float(state[0]), float(state[1])) for state in episode_states}))
        )
        trajectory_rows.extend(
            [float(episode), float(step), float(xy[0]), float(xy[1])]
            for step, xy in enumerate(episode_states)
        )
        episode_successes.append(float(result.terminal))

        elapsed_min = (time.monotonic() - started_at) / 60.0
        print(
            f"WWM-NAV episode={episode}/{config.episodes} "
            f"steps={int(env._step_count)} success={bool(result.terminal)} "
            f"return={episode_return:.3f} mean_outage={history['mean_outage'][-1]:.4f} "
            f"elapsed_min={elapsed_min:.1f}",
            flush=True,
        )

        checkpoint_due = (
            config.checkpoint_every_episodes > 0
            and (
                episode % int(config.checkpoint_every_episodes) == 0
                or episode == config.episodes
            )
        )
        if checkpoint_due:
            save_training_state(
                config.save_dir / "training_state.pt",
                {
                    "experiment_contract": EXPERIMENT_CONTRACT,
                    "episode": episode,
                    "global_step": global_step,
                    "config": _resume_config_payload(config),
                    "radiomap_semantic_sha256": semantic_npz_sha256(
                        config.radiomap_path
                    ),
                    "rssm_model": rssm_model.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "target_critic": target_critic.state_dict(),
                    "rssm_optimizer": rssm_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "return_norm": return_norm.state_dict(),
                    "advantage_norm": advantage_norm.state_dict(),
                    "replay": replay.state_dict(),
                    "history": history,
                    "training_tasks": training_tasks,
                    "trajectory_rows": trajectory_rows,
                    "episode_successes": episode_successes,
                    "actor_init_metrics": actor_init_metrics,
                    "critic_init_metrics": critic_init_metrics,
                    "rng_state": rng.bit_generator.state,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    ),
                },
            )

    save_training_artifacts(
        config,
        history,
        training_tasks,
        trajectory_rows,
        episode_successes,
        rssm_model,
        actor,
        critic,
        actor_init_metrics,
        critic_init_metrics,
    )
    return history


def save_training_artifacts(
    config: WWMNavConfig,
    history: dict[str, list[float]],
    training_tasks: list[tuple[int, float, float, float, float]],
    trajectory_rows: list[list[float]],
    episode_successes: list[float],
    rssm_model,
    actor,
    critic,
    actor_init_metrics: dict[str, float],
    critic_init_metrics: dict[str, float],
) -> None:
    config.save_dir.mkdir(parents=True, exist_ok=True)
    (config.save_dir / "config.json").write_text(
        json.dumps(serializable_config(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (config.save_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        keys = list(history)
        writer.writerow(["episode", *keys])
        for index in range(len(next(iter(history.values()), []))):
            writer.writerow([index + 1, *[history[key][index] for key in keys]])
    with (config.save_dir / "training_tasks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "start_x_m", "start_y_m", "goal_x_m", "goal_y_m"])
        writer.writerows(training_tasks)
    np.savez(
        config.save_dir / "episode_trajectories.npz",
        trajectory_points=np.asarray(trajectory_rows, dtype=np.float64),
        episode_success=np.asarray(episode_successes, dtype=np.float64),
        episode_start_m=np.asarray(
            [task[1:3] for task in training_tasks], dtype=np.float64
        ),
        episode_goal_m=np.asarray(
            [task[3:5] for task in training_tasks], dtype=np.float64
        ),
        episode_steps=np.asarray(history["episode_steps"], dtype=np.float64),
        mean_outage=np.asarray(history["mean_outage"], dtype=np.float64),
        max_outage=np.asarray(history["max_outage"], dtype=np.float64),
        high_outage_count=np.asarray(history["high_outage_count"], dtype=np.float64),
        high_outage_rate=np.asarray(history["high_outage_rate"], dtype=np.float64),
        measured_cells=np.asarray(history.get("measured_cells", []), dtype=np.float64),
    )
    plot_wwm_nav_training_history(
        history, config.save_dir / "wwm_nav_training.png"
    )
    plot_wwm_nav_world_model_training(
        history, config.save_dir / "wwm_nav_world_model_training.png"
    )
    plot_wwm_nav_actor_critic_training(
        history, config.save_dir / "wwm_nav_actor_critic_training.png"
    )
    print(
        "WWM-NAV training plots saved: "
        "wwm_nav_training.png, wwm_nav_world_model_training.png, "
        "wwm_nav_actor_critic_training.png",
        flush=True,
    )
    provenance = {
        "experiment_contract": EXPERIMENT_CONTRACT,
        "config_sha256": file_sha256(config.save_dir / "config.json"),
        "radiomap": str(config.radiomap_path),
        "radiomap_archive_sha256": file_sha256(config.radiomap_path),
        "radiomap_semantic_sha256": semantic_npz_sha256(config.radiomap_path),
        "training_tasks_sha256": file_sha256(config.save_dir / "training_tasks.csv"),
        "episode_trajectories_sha256": file_sha256(
            config.save_dir / "episode_trajectories.npz"
        ),
        "actor_navigation_prior_initialization": actor_init_metrics,
        "critic_navigation_prior_initialization": critic_init_metrics,
        **runtime_provenance(resolve_device(config.device)),
        **source_code_provenance(),
    }
    (config.save_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "experiment_contract": EXPERIMENT_CONTRACT,
            "rssm_model": rssm_model.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "config": serializable_config(config),
            "provenance": provenance,
        },
        config.save_dir / "checkpoint.pt",
    )


def evaluate(
    config: WWMNavConfig,
    checkpoint_path: Path | None = None,
    deterministic: bool = True,
) -> dict[str, object]:
    validate_wwm_nav_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    radio_map = RadioMap.from_npz(config.radiomap_path, require_physics=True)
    cache = PhysicalContextCache(radio_map, config)
    rssm_model, actor, _, _ = initialize_models(radio_map, config, device)
    path = checkpoint_path or (config.save_dir / "checkpoint.pt")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_contract") != EXPERIMENT_CONTRACT:
        raise ValueError("checkpoint is not a compatible WWM-NAV model")
    checkpoint_map_identity = checkpoint.get("provenance", {}).get(
        "radiomap_semantic_sha256"
    )
    if checkpoint_map_identity != semantic_npz_sha256(config.radiomap_path):
        raise ValueError("checkpoint radio-map identity does not match the evaluation map")
    rssm_model.load_state_dict(checkpoint["rssm_model"])
    actor.load_state_dict(checkpoint["actor"])
    rollout = evaluate_models(
        config, radio_map, rssm_model, actor, device, cache, deterministic=deterministic
    )
    config.save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(config.save_dir / "eval_rollout.npz", **rollout)
    summary = rollout_summary(
        rollout, np.asarray(config.goal_m, dtype=np.float64), config.high_outage_threshold
    )
    (config.save_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rollout
