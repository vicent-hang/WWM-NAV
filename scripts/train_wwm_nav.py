from __future__ import annotations

import argparse
from pathlib import Path

from wwm_nav.agent import WWMNavConfig, train
from wwm_nav.cli_config import load_config_file, merge_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WWM-NAV on a radio map.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--radiomap", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--start", nargs=2, type=float, default=None)
    parser.add_argument("--goal", nargs=2, type=float, default=None)
    parser.add_argument("--train-goal-mode", choices=("fixed", "random_starts_fixed_goal"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--actor-imagine-horizon", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--actor-batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_values = load_config_file(args.config) if args.config is not None else {}
    overrides = {
        "radiomap_path": args.radiomap,
        "save_dir": args.save_dir,
        "resume_checkpoint_path": args.resume,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "start_m": tuple(args.start) if args.start is not None else None,
        "goal_m": tuple(args.goal) if args.goal is not None else None,
        "train_goal_mode": args.train_goal_mode,
        "seed": args.seed,
        "device": args.device,
        "actor_imagine_horizon": args.actor_imagine_horizon,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "actor_batch_size": args.actor_batch_size,
        "checkpoint_every_episodes": args.checkpoint_every_episodes,
    }
    config_values = merge_config(config_values, overrides)
    if "radiomap_path" not in config_values:
        raise SystemExit("--radiomap or --config with radiomap is required")
    train(WWMNavConfig(**config_values))


if __name__ == "__main__":
    main()
