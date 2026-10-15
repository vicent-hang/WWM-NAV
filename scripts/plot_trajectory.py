
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wwm_nav.radiomap import RadioMap
from wwm_nav.tools.plotting import plot_path_on_radiomap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a WWM-NAV evaluation rollout.")
    parser.add_argument("--radiomap", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/c28_5_wwm_nav_seed1/eval_trajectory.png"),
    )
    parser.add_argument("--start", nargs=2, type=float, default=(100.0, 100.0))
    parser.add_argument("--goal", nargs=2, type=float, default=(1400.0, 1600.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radio_map = RadioMap.from_npz(args.radiomap, require_physics=False)
    rollout = np.load(args.rollout)
    plot_path_on_radiomap(
        radio_map,
        rollout["states"],
        np.asarray(args.start, dtype=float),
        np.asarray(args.goal, dtype=float),
        args.output,
    )


if __name__ == "__main__":
    main()
