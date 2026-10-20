from __future__ import annotations

import argparse
from pathlib import Path

from wwm_nav.radiomap import RadioMap
from wwm_nav.tools.plotting import plot_radio_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a standalone WWM-NAV outage-probability radio map."
    )
    parser.add_argument(
        "--radiomap",
        type=Path,
        default=Path("data/wwm_nav/radioenvir_wwm_nav.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wwm_nav/radioenvir_wwm_nav.png"),
    )
    parser.add_argument(
        "--title",
        default="WWM-NAV outage-probability radio map",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be >= 1")
    radio_map = RadioMap.from_npz(args.radiomap, require_physics=False)
    plot_radio_map(
        radio_map,
        args.output,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"saved WWM-NAV outage-probability map: {args.output}")


if __name__ == "__main__":
    main()
