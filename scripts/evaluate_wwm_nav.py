from __future__ import annotations

import argparse
from pathlib import Path

from wwm_nav.agent import WWMNavConfig, evaluate
from wwm_nav.cli_config import load_config_file, merge_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained WWM-NAV checkpoint.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--radiomap", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--start", nargs=2, type=float, default=None)
    parser.add_argument("--goal", nargs=2, type=float, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_values = load_config_file(args.config) if args.config is not None else {}
    overrides = {
        "radiomap_path": args.radiomap,
        "save_dir": args.save_dir,
        "max_steps": args.max_steps,
        "start_m": tuple(args.start) if args.start is not None else None,
        "goal_m": tuple(args.goal) if args.goal is not None else None,
        "device": args.device,
    }
    config_values = merge_config(config_values, overrides)
    if "radiomap_path" not in config_values:
        raise SystemExit("--radiomap or --config with radiomap is required")
    rollout = evaluate(WWMNavConfig(**config_values), checkpoint_path=args.checkpoint)
    mean_outage = float(rollout["outages"].mean()) if rollout["outages"].size else 0.0
    print(f"success={rollout['success']} steps={rollout['steps']} mean_outage={mean_outage:.4f}")


if __name__ == "__main__":
    main()
