from __future__ import annotations

import numpy as np
import torch

from wwm_nav.agent import EpisodeSequenceReplay


def test_episode_sequence_replay_samples_shapes():
    replay = EpisodeSequenceReplay(capacity_steps=100)
    obs0 = np.zeros(11, dtype=np.float32)
    replay.start_episode(obs0, np.asarray([0.0, 0.0]), np.asarray([10.0, 10.0]))
    for step in range(6):
        replay.append(
            action=step % 4,
            reward=1.0,
            outage=0.2,
            next_obs=np.ones(11, dtype=np.float32) * (step + 1),
            next_state_xy_m=np.asarray([float(step + 1), 0.0]),
            done=False,
        )
    batch = replay.sample_batch(2, 4, np.random.default_rng(0), torch.device("cpu"))
    assert batch["observations"].shape == (2, 5, 11)
    assert batch["actions"].shape == (2, 4)
    assert batch["state_xy_m"].shape == (2, 5, 2)
