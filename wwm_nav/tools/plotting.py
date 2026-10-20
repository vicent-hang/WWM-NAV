
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wwm_nav.radiomap import RadioMap


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wwm-nav")
    import matplotlib

    matplotlib.use("Agg")


def plot_path_on_radiomap(
    radio_map: RadioMap,
    path_m: np.ndarray,
    start_m: np.ndarray,
    goal_m: np.ndarray,
    output_path: Path,
    title: str | None = None,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    mesh = ax.contourf(
        radio_map.x_m,
        radio_map.y_m,
        radio_map.outage.T,
        levels=20,
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="Outage probability")
    ax.plot(
        path_m[:, 0],
        path_m[:, 1],
        color="white",
        linewidth=2.0,
        label="trajectory",
    )
    ax.plot(start_m[0], start_m[1], "rx", markersize=8, label="start")
    ax.plot(goal_m[0], goal_m[1], "b^", markersize=8, label="goal")
    if title:
        ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_episode_series(
    axis,
    values: object,
    label: str,
    *,
    color: str | None = None,
    rolling_window: int = 50,
) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return

    episodes = np.arange(1, array.size + 1)
    raw_line = axis.plot(
        episodes,
        array,
        linewidth=0.8,
        alpha=0.45,
        label=label,
        color=color,
    )[0]
    finite = np.isfinite(array)
    if array.size < rolling_window or not np.any(finite):
        return

    kernel = np.ones(rolling_window, dtype=np.float64)
    numerator = np.convolve(np.where(finite, array, 0.0), kernel, mode="valid")
    denominator = np.convolve(finite.astype(np.float64), kernel, mode="valid")
    rolling_mean = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    axis.plot(
        np.arange(rolling_window, array.size + 1),
        rolling_mean,
        linewidth=1.6,
        label=f"{label} rolling mean ({rolling_window})",
        color=raw_line.get_color(),
    )


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
        (
            axes[2, 0],
            "rssm_prior_outage_mae",
            "RSSM prior outage MAE",
            "RSSM prior outage MAE",
        ),
        (axes[2, 1], "measured_cells", "Measured radio cells", "cells"),
    )
    for axis, key, title, label in panels:
        _plot_episode_series(axis, history.get(key, []), label)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
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
        (axes[0, 0], "wm_loss", "RSSM training loss", "RSSM loss"),
        (
            axes[0, 1],
            "rssm_prior_outage_mae",
            "RSSM prior outage MAE",
            "replay prior outage MAE",
        ),
        (
            axes[1, 0],
            "rssm_ready_rate",
            "RSSM availability",
            "RSSM ready rate",
        ),
        (
            axes[1, 1],
            "imagined_predicted_env_reward_mean",
            "Imagined reward-head output",
            "predicted reward",
        ),
    ):
        _plot_episode_series(axis, history.get(key, []), label)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    axes[1, 0].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_wwm_nav_actor_critic_training(history: dict[str, list[float]], output_path: Path) -> None:
    """Save Dreamer-style Actor-Critic optimization diagnostics."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=160)
    fig.suptitle("WWM-NAV Actor-Critic training")
    _plot_episode_series(axes[0, 0], history.get("actor_loss", []), "actor loss")
    _plot_episode_series(
        axes[0, 0], history.get("actor_base_loss", []), "actor base loss"
    )
    axes[0, 0].set_title("Actor loss")
    _plot_episode_series(axes[0, 1], history.get("critic_loss", []), "critic loss")
    axes[0, 1].set_title("Critic loss")
    _plot_episode_series(
        axes[1, 0], history.get("actor_entropy", []), "actor entropy"
    )
    axes[1, 0].set_title("Actor entropy")
    _plot_episode_series(
        axes[1, 1], history.get("imagined_return_mean", []), "imagined return"
    )
    _plot_episode_series(
        axes[1, 1], history.get("imagined_reward_mean", []), "imagined reward"
    )
    axes[1, 1].set_title("Imagined targets")
    _plot_episode_series(
        axes[2, 0],
        history.get("actor_nav_prior_weight", []),
        "navigation-prior weight",
    )
    _plot_episode_series(
        axes[2, 0],
        history.get("actor_nav_prior_match_rate", []),
        "navigation-prior match rate",
    )
    axes[2, 0].set_title("Navigation prior")
    _plot_episode_series(
        axes[2, 1], history.get("actor_ready_rate", []), "actor ready rate"
    )
    axes[2, 1].set_title("Actor availability")
    axes[2, 1].set_ylim(-0.02, 1.02)
    for axis in axes.ravel():
        axis.set_xlabel("episode")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
