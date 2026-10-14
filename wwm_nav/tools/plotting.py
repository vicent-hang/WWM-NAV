
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
