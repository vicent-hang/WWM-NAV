"""Evaluate one WWM-NAV checkpoint on shared multi-step navigation trials."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wwm_nav.agent import (
    EXPERIMENT_CONTRACT,
    PhysicalContextCache,
    WWMNavConfig,
    evaluate_models,
    initialize_models,
    resolve_device,
    set_seed,
    validate_wwm_nav_config,
)
from wwm_nav.cli_config import load_config_file, merge_config
from wwm_nav.radiomap import RadioMap
from wwm_nav.wwm_nav_map import semantic_npz_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one WWM-NAV checkpoint on shared navigation trials."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--radiomap", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start", nargs=2, type=float, default=None)
    parser.add_argument("--goal", nargs=2, type=float, default=None)
    parser.add_argument("--min-initial-distance", type=float, default=800.0)
    parser.add_argument("--sample-margin", type=float, default=50.0)
    parser.add_argument("--max-sample-attempts", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--high-outage-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot-all", action="store_true")
    return parser.parse_args()


def read_trial_pairs(path: Path, radio_map: RadioMap) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"start_x", "start_y", "goal_x", "goal_y"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"pairs file is missing columns: {sorted(missing)}")
        for row in reader:
            start = np.asarray([float(row["start_x"]), float(row["start_y"])], dtype=np.float64)
            goal = np.asarray([float(row["goal_x"]), float(row["goal_y"])], dtype=np.float64)
            pairs.append(
                (
                    radio_map.index_to_xy(radio_map.nearest_index(start)),
                    radio_map.index_to_xy(radio_map.nearest_index(goal)),
                )
            )
    if not pairs:
        raise ValueError(f"pairs file is empty: {path}")
    return pairs


def sample_random_start_fixed_goal_pairs(
    radio_map: RadioMap,
    num_trials: int,
    seed: int,
    goal_m: np.ndarray,
    min_initial_distance: float,
    margin_m: float,
    max_attempts: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if max_attempts < num_trials:
        raise ValueError("max_sample_attempts must be >= num_trials")
    rng = np.random.default_rng(int(seed))
    goal = radio_map.index_to_xy(radio_map.nearest_index(goal_m))
    x_min, x_max, y_min, y_max = radio_map.bounds_m
    valid_x = radio_map.x_m[(radio_map.x_m >= x_min + margin_m) & (radio_map.x_m <= x_max - margin_m)]
    valid_y = radio_map.y_m[(radio_map.y_m >= y_min + margin_m) & (radio_map.y_m <= y_max - margin_m)]
    if valid_x.size == 0 or valid_y.size == 0:
        raise ValueError("sample margin leaves no valid map cells")
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[float, float]] = set()
    attempts = 0
    while len(pairs) < num_trials and attempts < max_attempts:
        attempts += 1
        start = np.asarray([rng.choice(valid_x), rng.choice(valid_y)], dtype=np.float64)
        key = (float(start[0]), float(start[1]))
        if key in seen or np.linalg.norm(start - goal) < min_initial_distance:
            continue
        seen.add(key)
        pairs.append((start, goal.copy()))
    if len(pairs) != num_trials:
        raise RuntimeError(
            f"sampled {len(pairs)} starts after {attempts} attempts; "
            "adjust --min-initial-distance, --sample-margin, or --max-sample-attempts"
        )
    return pairs


def save_trial_pairs(path: Path, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("start_x", "start_y", "goal_x", "goal_y"))
        writer.writeheader()
        for start, goal in pairs:
            writer.writerow({"start_x": start[0], "start_y": start[1], "goal_x": goal[0], "goal_y": goal[1]})


def count_reverse_actions(actions: np.ndarray) -> int:
    opposite = {0: 2, 1: 3, 2: 0, 3: 1}
    return sum(
        opposite.get(int(previous)) == int(current)
        for previous, current in zip(actions[:-1], actions[1:], strict=True)
    )


def count_repeated_states(states: np.ndarray) -> int:
    keys = [tuple(np.round(state, 6)) for state in states]
    return len(keys) - len(set(keys))


def trial_metrics(
    rollout: dict[str, object],
    trial: int,
    start_m: np.ndarray,
    goal_m: np.ndarray,
    max_steps: int,
    threshold: float,
) -> dict[str, Any]:
    states = np.asarray(rollout["states"], dtype=np.float64)
    actions = np.asarray(rollout["actions"], dtype=np.int64)
    rewards = np.asarray(rollout["rewards"], dtype=np.float64)
    outages = np.asarray(rollout["outages"], dtype=np.float64)
    terminals = np.asarray(rollout["terminals"], dtype=np.bool_)
    success = bool(terminals[-1]) if terminals.size else False
    steps = int(actions.size)
    path_length = float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum()) if len(states) > 1 else 0.0
    initial_distance = float(np.linalg.norm(goal_m - start_m))
    high_count = int(np.count_nonzero(outages >= threshold)) if outages.size else 0
    return {
        "trial": int(trial),
        "start_x": float(start_m[0]), "start_y": float(start_m[1]),
        "goal_x": float(goal_m[0]), "goal_y": float(goal_m[1]),
        "initial_distance_m": initial_distance,
        "success": int(success),
        "termination": "success" if success else ("max_steps" if steps >= max_steps else "outbound_or_other"),
        "steps": steps,
        "return": float(rewards.sum()) if rewards.size else 0.0,
        "mean_outage": float(outages.mean()) if outages.size else 0.0,
        "max_outage": float(outages.max()) if outages.size else 0.0,
        "cumulative_outage": float(outages.sum()) if outages.size else 0.0,
        "high_outage_threshold": float(threshold),
        "high_outage_count": high_count,
        "high_outage_rate": float(high_count / max(steps, 1)),
        "final_distance_m": float(np.linalg.norm(states[-1] - goal_m)) if states.size else float("nan"),
        "path_length_m": path_length,
        "path_efficiency": float(initial_distance / path_length) if path_length > 0.0 else 0.0,
        "reverse_count": count_reverse_actions(actions),
        "reverse_rate": float(count_reverse_actions(actions) / max(steps - 1, 1)),
        "repeat_count": count_repeated_states(states),
        "repeat_state_rate": float(count_repeated_states(states) / max(steps, 1)),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return float(np.mean([float(row[key]) for row in rows])) if rows else None


def _std(rows: list[dict[str, Any]], key: str) -> float | None:
    return float(np.std([float(row[key]) for row in rows])) if rows else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    return float(np.median([float(row[key]) for row in rows])) if rows else None


def summarize_trials(rows: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    successful = [row for row in rows if int(row["success"]) == 1]
    failed = [row for row in rows if int(row["success"]) == 0]
    return {
        "setting": settings,
        "num_trials": len(rows),
        "success_count": len(successful),
        "failure_count": len(failed),
        "success_rate": float(len(successful) / max(len(rows), 1)),
        "mean_all_steps": _mean(rows, "steps"), "std_all_steps": _std(rows, "steps"),
        "mean_all_outage": _mean(rows, "mean_outage"), "std_all_outage": _std(rows, "mean_outage"),
        "max_all_outage": max((float(row["max_outage"]) for row in rows), default=None),
        "mean_all_max_outage": _mean(rows, "max_outage"),
        "mean_all_final_distance_m": _mean(rows, "final_distance_m"), "median_all_final_distance_m": _median(rows, "final_distance_m"),
        "mean_all_path_efficiency": _mean(rows, "path_efficiency"),
        "mean_high_outage_count": _mean(rows, "high_outage_count"), "mean_high_outage_rate": _mean(rows, "high_outage_rate"),
        "max_high_outage_count": max((int(row["high_outage_count"]) for row in rows), default=None),
        "mean_success_steps": _mean(successful, "steps"), "std_success_steps": _std(successful, "steps"),
        "mean_success_outage": _mean(successful, "mean_outage"), "std_success_outage": _std(successful, "mean_outage"),
        "max_success_outage": max((float(row["max_outage"]) for row in successful), default=None),
        "mean_success_max_outage": _mean(successful, "max_outage"), "mean_success_cumulative_outage": _mean(successful, "cumulative_outage"),
        "mean_success_high_outage_count": _mean(successful, "high_outage_count"),
        "mean_success_high_outage_rate": _mean(successful, "high_outage_rate"),
        "mean_success_final_distance_m": _mean(successful, "final_distance_m"), "mean_success_path_length_m": _mean(successful, "path_length_m"),
        "mean_success_path_efficiency": _mean(successful, "path_efficiency"),
        "mean_success_repeat_count": _mean(successful, "repeat_count"),
        "mean_success_reverse_count": _mean(successful, "reverse_count"),
        "mean_repeat_count": _mean(rows, "repeat_count"),
        "mean_repeat_state_rate": _mean(rows, "repeat_state_rate"),
        "mean_reverse_count": _mean(rows, "reverse_count"),
        "mean_reverse_rate": _mean(rows, "reverse_rate"),
        "failure_trials": [int(row["trial"]) for row in failed],
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    metric_order = (
        "num_trials",
        "success_count",
        "failure_count",
        "success_rate",
        "mean_success_steps",
        "std_success_steps",
        "mean_success_outage",
        "std_success_outage",
        "max_success_outage",
        "mean_success_max_outage",
        "mean_success_high_outage_count",
        "mean_success_high_outage_rate",
        "mean_success_cumulative_outage",
        "mean_success_final_distance_m",
        "mean_success_path_length_m",
        "mean_success_path_efficiency",
        "mean_success_repeat_count",
        "mean_success_reverse_count",
        "mean_all_steps",
        "std_all_steps",
        "mean_all_outage",
        "std_all_outage",
        "max_all_outage",
        "mean_all_max_outage",
        "mean_high_outage_count",
        "mean_high_outage_rate",
        "max_high_outage_count",
        "mean_all_final_distance_m",
        "median_all_final_distance_m",
        "mean_all_path_efficiency",
        "mean_repeat_count",
        "mean_repeat_state_rate",
        "mean_reverse_count",
        "mean_reverse_rate",
        "failure_trials",
    )
    lines = ["# WWM-NAV shared-trial evaluation", "", "## Setting", ""]
    for key, value in summary.get("setting", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(("", "## Summary", ""))
    for key in metric_order:
        lines.append(f"- `{key}`: `{summary.get(key)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(save_dir: Path, pairs: list[tuple[np.ndarray, np.ndarray]], rows: list[dict[str, Any]], rollouts: list[dict[str, object]], summary: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_trial_pairs(save_dir / "trial_pairs.csv", pairs)
    with (save_dir / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = {key: np.asarray([row[key] for row in rows]) for key, value in rows[0].items() if isinstance(value, (int, float, np.integer, np.floating))}
    np.savez(save_dir / "trial_metrics.npz", **numeric)
    rollout_dir = save_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for row, rollout in zip(rows, rollouts, strict=True):
        np.savez(
            rollout_dir / f"trial_{int(row['trial']):04d}.npz",
            **rollout,
            trial_index=np.asarray([row["trial"]], dtype=np.int64),
            start_m=np.asarray([row["start_x"], row["start_y"]], dtype=np.float64),
            goal_m=np.asarray([row["goal_x"], row["goal_y"]], dtype=np.float64),
        )
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(save_dir / "summary.md", summary)
    plot_trial_metrics(save_dir / "trial_metrics.png", rows, summary)


def plot_trial_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    from wwm_nav.tools.plotting import configure_matplotlib

    configure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    trials = np.asarray([row["trial"] for row in rows], dtype=np.float64)
    success = np.asarray([row["success"] for row in rows], dtype=np.float64)
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float64)
    mean_outage = np.asarray(
        [row["mean_outage"] for row in rows], dtype=np.float64
    )
    max_outage = np.asarray([row["max_outage"] for row in rows], dtype=np.float64)
    high_outage_count = np.asarray(
        [row["high_outage_count"] for row in rows], dtype=np.float64
    )
    final_distance = np.asarray(
        [row["final_distance_m"] for row in rows], dtype=np.float64
    )
    repeat_count = np.asarray(
        [row["repeat_count"] for row in rows], dtype=np.float64
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=160)
    fig.suptitle(
        "WWM-NAV shared-trial metrics "
        f"(success_rate={summary['success_rate']:.3f}, trials={summary['num_trials']})"
    )
    axes[0, 0].bar(
        ["success", "failure"],
        [summary["success_count"], summary["failure_count"]],
        color=["tab:blue", "tab:red"],
    )
    axes[0, 0].set_title("Outcome count")
    axes[0, 0].set_ylabel("trials")

    axes[0, 1].plot(trials, steps, marker="o", linewidth=1.0, markersize=3)
    axes[0, 1].set_title("Steps per trial")
    axes[0, 1].set_xlabel("trial")
    axes[0, 1].set_ylabel("steps")

    axes[0, 2].scatter(
        mean_outage,
        final_distance,
        c=success,
        cmap="coolwarm_r",
        s=28,
    )
    axes[0, 2].set_title("Mean outage vs final distance")
    axes[0, 2].set_xlabel("mean outage")
    axes[0, 2].set_ylabel("final distance (m)")

    axes[1, 0].hist(mean_outage, bins=20, alpha=0.75, label="mean")
    axes[1, 0].hist(max_outage, bins=20, alpha=0.55, label="max")
    axes[1, 0].set_title("Outage distribution")
    axes[1, 0].set_xlabel("outage probability")
    axes[1, 0].set_ylabel("trials")
    axes[1, 0].legend()

    axes[1, 1].plot(
        trials,
        high_outage_count,
        marker="o",
        linewidth=1.0,
        markersize=3,
    )
    threshold = float(summary["setting"]["high_outage_threshold"])
    axes[1, 1].set_title(f"High-outage step count (threshold={threshold:.3g})")
    axes[1, 1].set_xlabel("trial")
    axes[1, 1].set_ylabel("steps")

    axes[1, 2].plot(
        trials,
        repeat_count,
        marker="o",
        linewidth=1.0,
        markersize=3,
    )
    axes[1, 2].set_title("Repeated states")
    axes[1, 2].set_xlabel("trial")
    axes[1, 2].set_ylabel("count")

    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_all_trajectories(path: Path, radio_map: RadioMap, rows: list[dict[str, Any]], rollouts: list[dict[str, object]]) -> None:
    from wwm_nav.tools.plotting import configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    mesh = ax.contourf(radio_map.x_m, radio_map.y_m, radio_map.outage.T, levels=20, cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="Outage probability")
    success_label_added = False
    failure_label_added = False
    for row, rollout in zip(rows, rollouts, strict=True):
        states = np.asarray(rollout["states"], dtype=np.float64)
        if states.ndim != 2 or states.shape[0] == 0:
            continue
        success = bool(row["success"])
        label = None
        if success and not success_label_added:
            label = "successful trajectory"
            success_label_added = True
        elif not success and not failure_label_added:
            label = "failed trajectory"
            failure_label_added = True
        ax.plot(
            states[:, 0],
            states[:, 1],
            color="white" if success else "red",
            alpha=0.72 if success else 0.95,
            linewidth=1.0 if success else 2.8,
            zorder=2 if success else 4,
            label=label,
        )
        if not success:
            ax.plot(states[-1, 0], states[-1, 1], "rx", markersize=6, zorder=5)

    starts = np.asarray(
        [[row["start_x"], row["start_y"]] for row in rows], dtype=np.float64
    )
    ax.scatter(
        starts[:, 0],
        starts[:, 1],
        marker="*",
        color="cyan",
        edgecolors="black",
        s=70,
        linewidths=0.5,
        zorder=6,
        label="start",
    )
    goals = {
        (float(row["goal_x"]), float(row["goal_y"]))
        for row in rows
    }
    for goal_index, (goal_x, goal_y) in enumerate(sorted(goals)):
        ax.plot(
            goal_x,
            goal_y,
            marker="^",
            color="blue",
            markeredgecolor="white",
            markersize=8,
            linestyle="None",
            zorder=7,
            label="goal" if goal_index == 0 else None,
        )

    success_count = sum(int(row["success"]) for row in rows)
    ax.set_title(
        "WWM-NAV shared navigation trials "
        f"({success_count}/{len(rows)} successful)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    values = load_config_file(args.config)
    values = merge_config(values, {"radiomap_path": args.radiomap, "save_dir": args.save_dir, "device": args.device})
    config = WWMNavConfig(**values)
    validate_wwm_nav_config(config)
    radio_map = RadioMap.from_npz(config.radiomap_path, require_physics=True)
    device = resolve_device(config.device)
    set_seed(config.seed)
    checkpoint_path = args.checkpoint or (config.save_dir / "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("experiment_contract") != EXPERIMENT_CONTRACT:
        raise ValueError("checkpoint is not a compatible WWM-NAV checkpoint")
    if checkpoint.get("provenance", {}).get("radiomap_semantic_sha256") != semantic_npz_sha256(config.radiomap_path):
        raise ValueError("checkpoint radio-map identity does not match the evaluation map")
    rssm_model, actor, _, _ = initialize_models(radio_map, config, device)
    rssm_model.load_state_dict(checkpoint["rssm_model"])
    actor.load_state_dict(checkpoint["actor"])
    rssm_model.eval()
    actor.eval()
    pairs = read_trial_pairs(args.pairs_file, radio_map) if args.pairs_file else sample_random_start_fixed_goal_pairs(
        radio_map, args.num_trials, args.seed, np.asarray(args.goal if args.goal is not None else config.goal_m, dtype=np.float64),
        args.min_initial_distance, args.sample_margin, args.max_sample_attempts,
    )
    if args.pairs_file is not None and len(pairs) != args.num_trials:
        raise ValueError(f"pairs file contains {len(pairs)} trials, expected {args.num_trials}")
    max_steps = int(args.max_steps if args.max_steps is not None else config.max_steps)
    threshold = float(args.high_outage_threshold if args.high_outage_threshold is not None else config.high_outage_threshold)
    trial_config = replace(config, max_steps=max_steps, high_outage_threshold=threshold)
    cache = PhysicalContextCache(radio_map, trial_config)
    rows: list[dict[str, Any]] = []
    rollouts: list[dict[str, object]] = []
    for index, (start, goal) in enumerate(pairs):
        rollout = evaluate_models(replace(trial_config, start_m=tuple(start), goal_m=tuple(goal)), radio_map, rssm_model, actor, device, cache, deterministic=True)
        rows.append(trial_metrics(rollout, index, start, goal, max_steps, threshold))
        rollouts.append(rollout)
    settings = {"method": "WWM-NAV", "checkpoint": str(checkpoint_path), "trial_mode": "random_starts_fixed_goal" if args.pairs_file is None else "pairs_file", "num_trials": len(rows), "seed": args.seed, "min_initial_distance": args.min_initial_distance, "sample_margin": args.sample_margin, "max_steps": max_steps, "high_outage_threshold": threshold}
    summary = summarize_trials(rows, settings)
    output_dir = args.save_dir or (config.save_dir / "shared_trials")
    write_outputs(output_dir, pairs, rows, rollouts, summary)
    if args.plot_all:
        plot_all_trajectories(output_dir / "all_trajectories.png", radio_map, rows, rollouts)
    print(f"saved WWM-NAV trial evaluation: {output_dir}")
    print(f"trials={len(rows)} success_rate={summary['success_rate']:.4f}")


if __name__ == "__main__":
    main()
