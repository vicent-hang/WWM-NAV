from __future__ import annotations

import numpy as np
import torch

from wwm_nav.agent import (
    PhysicalContextCache,
    WWMNavConfig,
    add_physical_context_to_batch,
    initialize_models,
    physics_dreamer_v3_imagine,
)
from wwm_nav.radiomap import RadioMap


def test_rssm_forward_and_imagination_shapes(tiny_radiomap_path):
    device = torch.device("cpu")
    radio_map = RadioMap.from_npz(tiny_radiomap_path, coordinate_scale=1.0, require_physics=True)
    config = WWMNavConfig(
        radiomap_path=tiny_radiomap_path,
        physical_num_bs=2,
        building_patch_size=3,
        rssm_deter_dim=16,
        rssm_stoch_dim=4,
        rssm_hidden_dim=32,
        rssm_embed_dim=16,
        physical_context_embed_dim=8,
        building_patch_embed_dim=8,
        actor_hidden_dims=(32,),
        actor_imagine_horizon=3,
    )
    cache = PhysicalContextCache(radio_map, config)
    rssm_model, actor, critic, target_critic = initialize_models(radio_map, config, device)
    batch = {
        "observations": torch.zeros(2, 5, config.observation_dim),
        "actions": torch.zeros(2, 4, dtype=torch.long),
        "state_xy_m": torch.zeros(2, 5, 2),
        "goal_xy_m": torch.ones(2, 2) * 20.0,
    }
    batch = add_physical_context_to_batch(batch, radio_map, config, device, cache)
    post, prior, post_pred, prior_pred = rssm_model.observe_sequence(
        batch["observations"],
        batch["actions"],
        physical_contexts=batch["physical_contexts"],
        building_patches=batch["building_patches"],
        sample=False,
    )
    assert post_pred["outages"].shape == (2, 4)
    start_state = {key: value[:, -1] for key, value in post.items()}
    imagination = physics_dreamer_v3_imagine(
        rssm_model,
        actor,
        start_state,
        torch.zeros(2, 2),
        torch.ones(2, 2) * 20.0,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2),
        radio_map,
        config,
        device,
        cache,
    )
    assert imagination.actions.shape == (config.actor_imagine_horizon, 2)
    assert imagination.rewards.shape == (config.actor_imagine_horizon, 2)
