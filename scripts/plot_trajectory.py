
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
        default=Path("outputs/wwm_nav_seed1/eval_trajectory.png"),
    )
    parser.add_argument("--start", nargs=2, type=float, default=None)
    parser.add_argument("--goal", nargs=2, type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radio_map = RadioMap.from_npz(args.radiomap, require_physics=False)
    rollout = np.load(args.rollout)
    states = np.asarray(rollout["states"], dtype=np.float64)
    start_m = np.asarray(
        args.start if args.start is not None else rollout.get("start_m", states[0]),
        dtype=np.float64,
    )
    goal_m = np.asarray(
        args.goal if args.goal is not None else rollout.get("goal_m", (1400.0, 1600.0)),
        dtype=np.float64,
    )
    plot_path_on_radiomap(
        radio_map,
        states,
        start_m,
        goal_m,
        args.output,
        title="WWM-NAV evaluation trajectory",
    )


if __name__ == "__main__":
    main()
