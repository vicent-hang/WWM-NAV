
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small synthetic radio map for WWM-NAV smoke runs.")
    parser.add_argument("--output", type=Path, default=Path("data/sample_radiomap.npz"))
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = int(args.grid_size)
    x = np.linspace(0.0, 200.0, n)
    y = np.linspace(0.0, 200.0, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    centers = np.asarray([[45.0, 55.0], [125.0, 40.0], [150.0, 145.0], [70.0, 150.0]])
    outage = np.full((n, n), 0.55, dtype=np.float64)
    for cx, cy in centers:
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        outage -= 0.23 * np.exp(-d2 / (2.0 * 24.0**2))
    for cx, cy in np.asarray([[95.0, 105.0], [35.0, 120.0], [165.0, 70.0]]):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        outage += 0.30 * np.exp(-d2 / (2.0 * 16.0**2))
    outage += 0.03 * rng.standard_normal((n, n))
    outage = np.clip(outage, 0.02, 0.98)
    height_map = np.clip(30.0 + 25.0 * np.sin(xx / 17.0) * np.cos(yy / 23.0), 0.0, 90.0)
    bs_loc_km = np.asarray([[0.45, 0.55, 0.035], [1.25, 0.40, 0.035], [1.50, 1.45, 0.035], [0.70, 1.50, 0.035]])
    sector_boresite_deg = np.asarray([0.0, 120.0, 240.0])
    points = np.stack([xx, yy], axis=-1) * 10.0
    bs_m = bs_loc_km[:, :2] * 1000.0
    dist = np.linalg.norm(points[:, :, None, :] - bs_m[None, None, :, :], axis=-1)
    best_bs = np.argmin(dist, axis=-1).astype(np.int64)
    best_sector = np.zeros_like(best_bs, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        outage,
        x,
        y,
        bs_loc_km=bs_loc_km,
        sector_boresite_deg=sector_boresite_deg,
        bs_power_watt=np.asarray([10.0]),
        bs_power_watt_by_bs=np.full(bs_loc_km.shape[0], 10.0),
        active_original_bs_ids=np.arange(bs_loc_km.shape[0]),
        height_map=height_map,
        best_bs=best_bs,
        best_sector=best_sector,
        uav_height_km=np.asarray([0.1]),
        max_height_m=np.asarray([90.0]),
    )
    print(args.output)


if __name__ == "__main__":
    main()
