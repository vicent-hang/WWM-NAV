
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wwm_nav.radiomap import RadioMap


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wwm-nav")
    import matplotlib

    matplotlib.use("Agg")


def plot_path_on_radiomap(radio_map: RadioMap, path_m: np.ndarray, start_m: np.ndarray, goal_m: np.ndarray, output_path: Path, title: str = "WWM-NAV trajectory") -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    mesh = ax.contourf(radio_map.x_m, radio_map.y_m, radio_map.outage.T, levels=20, cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="Outage probability")
    ax.plot(path_m[:, 0], path_m[:, 1], color="tab:red", linewidth=2.5, label="WWM-NAV")
    ax.plot(start_m[0], start_m[1], "o", color="white", markeredgecolor="black", markersize=8, label="Start")
    ax.plot(goal_m[0], goal_m[1], "*", color="tab:red", markeredgecolor="white", markersize=13, label="Goal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_episode_series(axis, values: object, label: str) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size:
        axis.plot(np.arange(1, array.size + 1), array, linewidth=1.4, label=label)


def plot_wwm_nav_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Save the main online multi-step training dashboard."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=160)
    fig.suptitle("WWM-NAV online multi-step training")
    panels = (
        (axes[0, 0], "episode_return", "Episode return", "return"),
        (axes[0, 1], "success", "Success", "success"),
        (axes[1, 0], "mean_outage", "Mean outage", "mean outage"),
        (axes[1, 1], "episode_steps", "Episode steps", "steps"),
        (axes[2, 0], "rssm_prior_outage_mae", "RSSM prior outage MAE", "MAE"),
        (axes[2, 1], "measured_cells", "Measured radio cells", "cells"),
    )
    for axis, key, title, label in panels:
        _plot_episode_series(axis, history.get(key, []), label)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.grid(True, alpha=0.25)
        axis.legend()
    _plot_episode_series(axes[1, 0], history.get("high_outage_rate", []), "high outage rate")
    axes[1, 0].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_wwm_nav_world_model_training(history: dict[str, list[float]], output_path: Path) -> None:
    """Save RSSM loss and outage-prediction diagnostics."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    fig.suptitle("WWM-NAV RSSM world-model training")
    for axis, key, title, label in (
        (axes[0, 0], "wm_loss", "RSSM training loss", "loss"),
        (axes[0, 1], "rssm_prior_outage_mae", "RSSM prior outage MAE", "MAE"),
        (axes[1, 0], "mean_outage", "Measured mean outage", "outage"),
        (axes[1, 1], "high_outage_rate", "High-outage rate", "rate"),
    ):
        _plot_episode_series(axis, history.get(key, []), label)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_wwm_nav_actor_critic_training(history: dict[str, list[float]], output_path: Path) -> None:
    """Save Dreamer-style Actor-Critic optimization diagnostics."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    fig.suptitle("WWM-NAV Actor-Critic training")
    for axis, key, title, label in (
        (axes[0, 0], "actor_loss", "Actor loss", "actor loss"),
        (axes[0, 1], "critic_loss", "Critic loss", "critic loss"),
        (axes[1, 0], "imagined_return_mean", "Imagined return", "return"),
        (axes[1, 1], "episode_return", "Episode return", "return"),
    ):
        _plot_episode_series(axis, history.get(key, []), label)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
