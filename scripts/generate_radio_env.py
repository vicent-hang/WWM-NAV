from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wwm_nav.radio_env import SnarmLikeRadioEnvironment, SnarmRadioConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the C28.5 SNARM-style physical radio environment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/radioenvir_c28_5_generated.npz"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast-fading-samples", type=int, default=1000)
    parser.add_argument("--los-samples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SnarmRadioConfig(
        fast_fading_samples=int(args.fast_fading_samples),
        los_samples=int(args.los_samples),
    )
    environment = SnarmLikeRadioEnvironment(config=config, seed=int(args.seed))
    radio_data = environment.generate_radio_map_data()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        radio_data.outage,
        radio_data.x_vec,
        radio_data.y_vec,
        radio_data.test_loc_m,
        radio_data.test_outage,
        bs_loc_km=radio_data.bs_loc_km,
        sector_boresite_deg=radio_data.sector_boresite_deg,
        bs_power_watt=np.asarray([radio_data.bs_power_watt], dtype=np.float64),
        height_map=radio_data.height_map,
        best_bs=radio_data.best_bs,
        best_sector=radio_data.best_sector,
        uav_height_km=np.asarray([radio_data.uav_height_km], dtype=np.float64),
        max_height_m=np.asarray([radio_data.max_height_m], dtype=np.float64),
        bs_downtilt_deg=np.asarray([config.bs_downtilt_deg], dtype=np.float64),
        carrier_freq_ghz=np.asarray([config.carrier_freq_ghz], dtype=np.float64),
        los_samples=np.asarray([config.los_samples], dtype=np.int64),
    )
    print(f"saved C28.5 radio environment: {args.output}")
    print(f"outage shape: {radio_data.outage.shape}")
    print(
        "outage min/max: "
        f"{float(radio_data.outage.min()):.4f}/"
        f"{float(radio_data.outage.max()):.4f}"
    )


if __name__ == "__main__":
    main()
