from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wwm-nav")

import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from wwm_nav.radiomap import RadioMap
from wwm_nav.tools.magazine_data import (
    WWM_LA_NAV,
    load_method,
    read_provenance,
    validate_inputs,
)
from wwm_nav.tools.plotting import configure_matplotlib

OUTPUT_STEM = "panel_a_radio_map_navigation_trial0004"
DISPLAY_METHOD_NAME = "WWM-NAV"
PATH_COLOR = "#E64B35"
STYLE = {
    "font.size": 14.0,
    "axes.labelsize": 14.0,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.0,
    "figure.dpi": 180,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
TEXT_SCALE = 1.4


def scale_descriptive_text(fig: Any, factor: float) -> None:
    for axis in fig.axes:
        for text in (axis.title, axis.xaxis.label, axis.yaxis.label):
            text.set_fontsize(text.get_fontsize() * factor)
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(text.get_fontsize() * factor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a provenance-bound 2D WWM-NAV radio-map scene with the "
            "stored trajectory and a supplied UAV icon."
        )
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("reference/magazine/data")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs")
    )
    parser.add_argument(
        "--uav-icon", type=Path, default=Path("reference/magazine/icon/uav.png")
    )
    parser.add_argument("--representative-trial", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / f"{OUTPUT_STEM}.png",
        output_dir / f"{OUTPUT_STEM}.pdf",
        output_dir / f"{OUTPUT_STEM}.json",
    ]


def add_direction_arrows(axis: Any, states: np.ndarray) -> None:
    movement = np.linalg.norm(np.diff(states, axis=0), axis=1)
    moving_segments = np.flatnonzero(movement > 1e-9)
    if moving_segments.size == 0:
        return
    selected = np.unique(
        moving_segments[
            np.rint(np.linspace(0, moving_segments.size - 1, 5)).astype(np.int64)
        ]
    )
    for index in selected:
        axis.annotate(
            "",
            xy=states[index + 1],
            xytext=states[index],
            arrowprops={
                "arrowstyle": "-|>",
                "color": PATH_COLOR,
                "linewidth": 1.8,
                "mutation_scale": 14,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=7,
        )


def add_uav_icon(axis: Any, icon_path: Path, states: np.ndarray) -> None:
    icon = mpimg.imread(icon_path)
    movement = np.linalg.norm(np.diff(states, axis=0), axis=1)
    moving_states = np.flatnonzero(np.r_[True, movement > 1e-9])
    icon_index = int(moving_states[min(moving_states.size - 1, moving_states.size // 3)])
    artist = AnnotationBbox(
        OffsetImage(icon, zoom=0.17),
        states[icon_index],
        frameon=True,
        pad=0.12,
        bboxprops={
            "boxstyle": "circle,pad=0.12",
            "facecolor": "white",
            "edgecolor": "#163A63",
            "linewidth": 1.1,
            "alpha": 0.94,
        },
        zorder=9,
    )
    axis.add_artist(artist)


def plot_scene(
    radio_map: RadioMap,
    states: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    threshold: float,
    icon_path: Path,
) -> Any:
    fig, axis = plt.subplots(figsize=(7.2, 6.5))
    extent = [
        float(radio_map.x_m[0]),
        float(radio_map.x_m[-1]),
        float(radio_map.y_m[0]),
        float(radio_map.y_m[-1]),
    ]
    image = axis.imshow(
        radio_map.outage.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        zorder=0,
    )

    xx, yy = np.meshgrid(radio_map.x_m, radio_map.y_m, indexing="xy")
    axis.contour(
        xx,
        yy,
        radio_map.outage.T,
        levels=[threshold],
        colors="#202020",
        linestyles="--",
        linewidths=0.85,
        alpha=0.72,
        zorder=3,
    )

    (path_line,) = axis.plot(
        states[:, 0],
        states[:, 1],
        color=PATH_COLOR,
        linewidth=3.0,
        label=DISPLAY_METHOD_NAME,
        zorder=6,
    )
    path_line.set_path_effects(
        [
            path_effects.Stroke(linewidth=5.4, foreground="white"),
            path_effects.Stroke(linewidth=4.1, foreground="#3A3A3A"),
            path_effects.Normal(),
        ]
    )
    add_direction_arrows(axis, states)

    axis.scatter(
        start[0],
        start[1],
        s=78,
        marker="o",
        facecolor="#FFFFFF",
        edgecolor="#111111",
        linewidth=1.5,
        zorder=8,
    )
    axis.scatter(
        goal[0],
        goal[1],
        s=190,
        marker="*",
        facecolor="#D62728",
        edgecolor="white",
        linewidth=1.2,
        zorder=8,
    )
    add_uav_icon(axis, icon_path, states)

    x_minor = np.asarray(radio_map.x_m, dtype=np.float64)
    y_minor = np.asarray(radio_map.y_m, dtype=np.float64)
    major_stride_x = max(1, (x_minor.size - 1) // 10)
    major_stride_y = max(1, (y_minor.size - 1) // 10)
    x_major = x_minor[::major_stride_x]
    y_major = y_minor[::major_stride_y]
    if not np.isclose(x_major[-1], x_minor[-1]):
        x_major = np.append(x_major, x_minor[-1])
    if not np.isclose(y_major[-1], y_minor[-1]):
        y_major = np.append(y_major, y_minor[-1])
    axis.set_xticks(x_major)
    axis.set_yticks(y_major)
    axis.set_xticks(x_minor, minor=True)
    axis.set_yticks(y_minor, minor=True)
    axis.tick_params(which="minor", length=0)
    axis.grid(True, which="major", color="white", linewidth=1.30, alpha=0.90, zorder=4)
    axis.grid(True, which="minor", color="white", linewidth=0.42, alpha=0.45, zorder=4)
    axis.set_axisbelow(False)
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")

    legend_handles = [
        Line2D([0], [0], color=PATH_COLOR, linewidth=3.0, label=DISPLAY_METHOD_NAME),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor="black", markersize=7.5, label="Start",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="none", markerfacecolor="#D62728",
            markeredgecolor="white", markersize=12, label="Goal",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        framealpha=0.92,
        ncol=2,
        fontsize=10.0,
        columnspacing=0.9,
        handlelength=2.1,
    )
    colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", fraction=0.052, pad=0.10, aspect=34)
    colorbar.set_label("Outage probability")
    return fig


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    icon_path = args.uav_icon.resolve()
    if not icon_path.is_file():
        raise FileNotFoundError(icon_path)

    provenance = read_provenance(analysis_dir)
    radio_map_path, directories = validate_inputs(provenance)
    trial = int(args.representative_trial)
    if trial < 0 or trial >= int(provenance["num_trials"]):
        raise ValueError(f"representative trial must be in [0, {provenance['num_trials'] - 1}]")

    targets = output_paths(output_dir)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing scene outputs: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    radio_map = RadioMap.from_npz(radio_map_path)
    threshold = float(provenance["high_outage_threshold"])
    rows, rollouts, _ = load_method(
        WWM_LA_NAV,
        directories[WWM_LA_NAV],
        radio_map,
        threshold,
    )
    if not rows[trial]["success"]:
        raise ValueError(f"WWM-LA-NAV trial {trial:04d} did not reach the goal")
    rollout = rollouts[trial]

    with plt.rc_context(STYLE):
        fig = plot_scene(
            radio_map,
            np.asarray(rollout["states"], dtype=np.float64),
            np.asarray(rollout["start"], dtype=np.float64),
            np.asarray(rollout["goal"], dtype=np.float64),
            threshold,
            icon_path,
        )
        fig.tight_layout(pad=0.55)
        scale_descriptive_text(fig, TEXT_SCALE)
        fig.savefig(targets[0], dpi=320, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(targets[1], bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    manifest = {
        "artifact": OUTPUT_STEM,
        "analysis_contract": provenance["analysis_contract"],
        "method": DISPLAY_METHOD_NAME,
        "source_method_key": WWM_LA_NAV,
        "representative_trial": trial,
        "trajectory_source": str(directories[WWM_LA_NAV] / "rollouts" / f"trial_{trial:04d}.npz"),
        "radio_map_source": str(radio_map_path),
        "radio_map_sha256": provenance["radiomap_sha256"],
        "trial_metrics_sha256": provenance["trial_metrics_sha256"][WWM_LA_NAV],
        "uav_icon": str(icon_path),
        "high_outage_threshold": threshold,
        "action_grid_resolution_m": [
            float(radio_map.resolution_m[0]),
            float(radio_map.resolution_m[1]),
        ],
        "steps": int(rows[trial]["steps"]),
        "success": int(rows[trial]["success"]),
    }
    targets[2].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved provenance-bound WWM-NAV radio-map scene: {targets[0]}")


if __name__ == "__main__":
    main()
