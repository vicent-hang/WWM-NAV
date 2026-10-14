from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_radiomap_path(tmp_path: Path) -> Path:
    x = np.linspace(0.0, 20.0, 5)
    y = np.linspace(0.0, 20.0, 5)
    outage = np.full((5, 5), 0.2, dtype=np.float64)
    outage[2, 2] = 0.8
    height_map = np.zeros((5, 5), dtype=np.float64)
    bs_loc_km = np.asarray([[0.0, 0.0, 0.03], [0.02, 0.02, 0.03]], dtype=np.float64)
    best_bs = np.zeros((5, 5), dtype=np.int64)
    best_sector = np.zeros((5, 5), dtype=np.int64)
    path = tmp_path / "map.npz"
    np.savez(
        path,
        outage,
        x,
        y,
        bs_loc_km=bs_loc_km,
        sector_boresite_deg=np.asarray([0.0]),
        bs_power_watt=np.asarray([1.0]),
        bs_power_watt_by_bs=np.ones(bs_loc_km.shape[0]),
        active_original_bs_ids=np.arange(bs_loc_km.shape[0]),
        height_map=height_map,
        best_bs=best_bs,
        best_sector=best_sector,
        uav_height_km=np.asarray([0.1]),
        max_height_m=np.asarray([90.0]),
    )
    return path
