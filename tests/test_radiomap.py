from __future__ import annotations

import numpy as np

from wwm_nav.radiomap import RadioMap


def test_radiomap_loads_required_physics(tiny_radiomap_path):
    radio_map = RadioMap.from_npz(tiny_radiomap_path, coordinate_scale=1.0, require_physics=True)
    assert radio_map.outage.shape == (5, 5)
    assert radio_map.outage_at_xy([10.0, 10.0]) == 0.8
    context = radio_map.physical_context_at_xy(np.asarray([10.0, 10.0]), num_bs=2)
    assert context.shape == (RadioMap.physical_context_dim(2),)
    patch = radio_map.building_patch_at_xy([10.0, 10.0], patch_size=3)
    assert patch.shape == (1, 3, 3)
