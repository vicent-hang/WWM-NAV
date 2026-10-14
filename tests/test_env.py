from __future__ import annotations

import numpy as np

from wwm_nav.agent import WWMNavConfig, build_online_observation, make_env
from wwm_nav.radiomap import RadioMap


def test_grid_navigation_step_and_observation(tiny_radiomap_path):
    radio_map = RadioMap.from_npz(tiny_radiomap_path, coordinate_scale=1.0, require_physics=True)
    config = WWMNavConfig(
        radiomap_path=tiny_radiomap_path,
        start_m=(0.0, 0.0),
        goal_m=(20.0, 20.0),
        physical_num_bs=2,
        building_patch_size=3,
        max_steps=10,
    )
    env = make_env(radio_map, np.asarray(config.goal_m), config)
    state = env.reset(start_m=np.asarray(config.start_m))
    obs = build_online_observation(env, state, None, 0.0, config)
    assert obs.shape == (config.observation_dim,)
    result = env.step_action(1)
    assert result.state.shape == (2,)
    assert 0.0 <= result.outage <= 1.0
