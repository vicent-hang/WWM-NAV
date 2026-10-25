from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wwm-nav")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from wwm_nav.radiomap import RadioMap
from wwm_nav.tools.magazine_data import (
    COLORS,
    DISPLAY_NAMES,
    LINE_STYLES,
    MARKER_OFFSETS,
    MARKERS,
    METHODS,
    cumulative_outage_curves,
    goal_reaching_curve,
    load_method,
    read_provenance,
    validate_inputs,
    validate_pairing,
)
from wwm_nav.tools.plotting import configure_matplotlib

PANEL_STEMS = {
    "environment": "panel_a_wireless_environment_shared_tasks",
    "reaching": "panel_b_cumulative_goal_reaching",
    "cumulative_outage": "panel_c_stepwise_cumulative_outage",
    "trajectory": "panel_d_representative_trajectory_trial0004",
    "trial_outage": "panel_e_representative_outage_trial0004",
}
COMPOSITE_STEM = "wwm_nav_magazine_case_study"
MAGAZINE_DISPLAY_NAMES = {
    **DISPLAY_NAMES,
    "SNARM": "SNARM",
    "WWM-LA-NAV": "WWM-NAV",
}
PANEL_TITLES = {
    "environment": "(a) Wireless Environment and Shared Tasks",
    "reaching": "(b) Cumulative Goal-Reaching",
    "cumulative_outage": "(c) Cumulative Outage Exposure",
    "trajectory": "(d) Representative Paths (Trial 0004)",
    "trial_outage": "(e) Step-Wise Outage (Trial 0004)",
}
STANDALONE_STYLE = {
    "font.size": 14.0,
    "axes.titlesize": 14.5,
    "axes.labelsize": 14.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 11.5,
    "figure.dpi": 180,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
COMPOSITE_STYLE = {
    "font.size": 12.0,
    "axes.titlesize": 12.0,
    "axes.titlepad": 5.0,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 8.0,
    "figure.dpi": 180,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
TEXT_SCALE = 1.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the WWM-NAV magazine-style case-study composite and "
            "its five standalone panels from completed paired-trial artifacts."
        )
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("reference/magazine/data")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs")
    )
    parser.add_argument("--representative-trial", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(output_dir: Path) -> list[Path]:
    stems = [*PANEL_STEMS.values(), COMPOSITE_STEM]
    return [output_dir / f"{stem}.{suffix}" for stem in stems for suffix in ("png", "pdf")]


def scale_descriptive_text(fig: Any, factor: float) -> None:
    for axis in fig.axes:
        for text in (axis.title, axis.xaxis.label, axis.yaxis.label):
            text.set_fontsize(text.get_fontsize() * factor)
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(text.get_fontsize() * factor)


def save_figure(fig: Any, output_dir: Path, stem: str, *, composite: bool = False) -> None:
    if composite:
        fig.subplots_adjust(left=0.045, right=0.985, top=0.935, bottom=0.105, wspace=0.27, hspace=0.42)
    else:
        fig.tight_layout(pad=0.65)
    scale_descriptive_text(fig, TEXT_SCALE)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def style_axis(axis: Any) -> None:
    axis.grid(True, color="0.85", linewidth=0.75, alpha=0.75, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def method_curve(
    axis: Any,
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    markevery: int,
    drawstyle: str = "default",
    linewidth: float = 2.2,
) -> Any:
    (line,) = axis.plot(
        x,
        y,
        color=COLORS[method],
        linestyle=LINE_STYLES[method],
        linewidth=linewidth,
        marker=MARKERS[method],
        markevery=(MARKER_OFFSETS[method], markevery),
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=COLORS[method],
        markeredgewidth=1.0,
        drawstyle=drawstyle,
        label=MAGAZINE_DISPLAY_NAMES[method],
        zorder=4,
    )
    return line


def plot_environment(
    axis: Any,
    fig: Any,
    radio_map: RadioMap,
    tables: dict[str, list[dict[str, Any]]],
    threshold: float,
    *,
    title: bool,
    compact: bool = False,
) -> None:
    extent = [radio_map.x_m[0], radio_map.x_m[-1], radio_map.y_m[0], radio_map.y_m[-1]]
    image = axis.imshow(
        radio_map.outage.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=0,
    )
    xx, yy = np.meshgrid(radio_map.x_m, radio_map.y_m, indexing="xy")
    axis.contour(
        xx,
        yy,
        radio_map.outage.T,
        levels=[threshold],
        colors="white",
        linestyles="--",
        linewidths=1.2,
        zorder=2,
    )
    reference_rows = tables[METHODS[0]]
    starts = np.asarray([[row["start_x"], row["start_y"]] for row in reference_rows])
    goal = np.asarray([reference_rows[0]["goal_x"], reference_rows[0]["goal_y"]])
    axis.scatter(
        starts[:, 0],
        starts[:, 1],
        s=18,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.82,
        label="Shared starts (N = 100)",
        zorder=4,
    )
    axis.scatter(
        goal[0],
        goal[1],
        s=150,
        marker="*",
        facecolor="#E41A1C",
        edgecolor="white",
        linewidth=1.1,
        label="Fixed goal",
        zorder=6,
    )
    if radio_map.bs_loc_km is not None:
        bs_xy_m = np.asarray(radio_map.bs_loc_km[:, :2], dtype=np.float64) * 1000.0
        axis.scatter(
            bs_xy_m[:, 0],
            bs_xy_m[:, 1],
            s=55,
            marker="^",
            facecolor="#FFDD55",
            edgecolor="black",
            linewidth=0.8,
            label="Base station",
            zorder=5,
        )
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="white", linestyle="--", linewidth=1.3))
    labels.append(f"Outage = {threshold:.2f}")
    axis.legend(
        handles,
        labels,
        loc="lower left",
        framealpha=0.90,
        ncol=2,
        fontsize=7.6 if compact else None,
        handlelength=2.0 if compact else None,
        columnspacing=0.8 if compact else None,
        labelspacing=0.3 if compact else None,
    )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="box")
    if title:
        axis.set_title(PANEL_TITLES["environment"], fontweight="bold")
    colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", fraction=0.045, pad=0.085, aspect=35)
    colorbar.set_label("Outage probability")


def plot_reaching(
    axis: Any,
    tables: dict[str, list[dict[str, Any]]],
    max_steps: int,
    *,
    title: bool,
    compact: bool = False,
) -> None:
    x = np.arange(max_steps + 1)
    indices = np.arange(len(tables[METHODS[0]]), dtype=np.int64)
    for method in METHODS:
        curve = goal_reaching_curve(tables[method], indices, max_steps)
        method_curve(axis, method, x, curve, markevery=35, drawstyle="steps-post")
    axis.set_xlabel("Step budget")
    axis.set_ylabel("Goal-reaching rate")
    axis.set_xlim(0, max_steps)
    axis.set_ylim(0.0, 1.04)
    style_axis(axis)
    axis.legend(
        loc="lower right",
        framealpha=0.9,
        fontsize=8.0 if compact else None,
        handlelength=2.2 if compact else None,
        labelspacing=0.3 if compact else None,
    )
    if title:
        axis.set_title(PANEL_TITLES["reaching"], fontweight="bold")


def plot_cumulative_outage(
    axis: Any,
    rollouts: dict[str, list[dict[str, np.ndarray]]],
    max_steps: int,
    *,
    title: bool,
    compact: bool = False,
) -> None:
    x = np.arange(max_steps + 1)
    for method in METHODS:
        curves = cumulative_outage_curves(rollouts[method], max_steps)
        method_curve(axis, method, x, np.mean(curves, axis=0), markevery=35)
    axis.set_xlabel("Executed step")
    axis.set_ylabel("Mean cumulative outage")
    axis.set_xlim(0, max_steps)
    axis.set_ylim(bottom=0.0)
    style_axis(axis)
    axis.legend(
        loc="upper left",
        framealpha=0.9,
        fontsize=8.0 if compact else None,
        handlelength=2.2 if compact else None,
        labelspacing=0.3 if compact else None,
    )
    if title:
        axis.set_title(PANEL_TITLES["cumulative_outage"], fontweight="bold")


def plot_representative_trajectory(
    axis: Any,
    radio_map: RadioMap,
    rollouts: dict[str, list[dict[str, np.ndarray]]],
    trial: int,
    *,
    title: bool,
    compact: bool = False,
) -> None:
    extent = [radio_map.x_m[0], radio_map.x_m[-1], radio_map.y_m[0], radio_map.y_m[-1]]
    axis.imshow(
        radio_map.outage.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.78,
        interpolation="nearest",
        zorder=0,
    )
    for method in METHODS:
        states = np.asarray(rollouts[method][trial]["states"], dtype=np.float64)
        line = method_curve(axis, method, states[:, 0], states[:, 1], markevery=20, linewidth=2.0)
        line.set_path_effects(
            [
                path_effects.Stroke(linewidth=4.0, foreground="white"),
                path_effects.Stroke(linewidth=2.9, foreground="black"),
                path_effects.Normal(),
            ]
        )
    reference = rollouts[METHODS[0]][trial]
    axis.scatter(*reference["start"], marker="*", s=100, color="black", label="Start", zorder=7)
    axis.scatter(*reference["goal"], marker="x", s=80, color="black", linewidth=2.0, label="Goal", zorder=7)
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(
        loc="upper left" if compact else "lower right",
        framealpha=0.90,
        ncol=2 if compact else 1,
        fontsize=7.2 if compact else None,
        handlelength=2.1 if compact else None,
        columnspacing=0.7 if compact else None,
        labelspacing=0.25 if compact else None,
        borderpad=0.35 if compact else None,
    )
    if title:
        axis.set_title(PANEL_TITLES["trajectory"], fontweight="bold")


def plot_representative_outage(
    axis: Any,
    rollouts: dict[str, list[dict[str, np.ndarray]]],
    trial: int,
    threshold: float,
    *,
    title: bool,
    compact: bool = False,
) -> None:
    max_trial_steps = 0
    for method in METHODS:
        outages = np.asarray(rollouts[method][trial]["outages"], dtype=np.float64)
        steps = np.arange(1, outages.size + 1)
        method_curve(axis, method, steps, outages, markevery=20, linewidth=1.9)
        max_trial_steps = max(max_trial_steps, int(outages.size))
    axis.axhline(threshold, color="black", linestyle=":", linewidth=1.2, label="Threshold")
    axis.set_xlabel("Executed step")
    axis.set_ylabel("Outage probability")
    axis.set_xlim(0, max_trial_steps)
    axis.set_ylim(0.0, 1.0)
    style_axis(axis)
    axis.legend(
        loc="upper right",
        framealpha=0.90,
        ncol=2 if compact else 1,
        fontsize=7.4 if compact else None,
        handlelength=2.1 if compact else None,
        columnspacing=0.7 if compact else None,
        labelspacing=0.25 if compact else None,
        borderpad=0.35 if compact else None,
    )
    if title:
        axis.set_title(PANEL_TITLES["trial_outage"], fontweight="bold")


def standalone_panel(
    output_dir: Path,
    stem: str,
    figsize: tuple[float, float],
    plotter: Callable[[Any, Any], None],
) -> None:
    fig, axis = plt.subplots(figsize=figsize)
    plotter(axis, fig)
    save_figure(fig, output_dir, stem)


def add_magazine_separators(fig: Any, axes: tuple[Any, Any, Any, Any, Any]) -> None:
    fig.canvas.draw()
    env, reaching, cumulative, trajectory, trial_outage = axes
    env_box = env.get_position()
    reaching_box = reaching.get_position()
    cumulative_box = cumulative.get_position()
    trajectory_box = trajectory.get_position()
    trial_box = trial_outage.get_position()
    separator_color = "#496DAA"
    x_first = (env_box.x1 + reaching_box.x0) / 2.0
    x_second = (reaching_box.x1 + trajectory_box.x0) / 2.0
    y_middle = (reaching_box.y0 + cumulative_box.y1) / 2.0
    for x in (x_first, x_second):
        fig.add_artist(
            Line2D(
                [x, x], [0.055, 0.965], transform=fig.transFigure,
                color=separator_color, linestyle=(0, (6, 4)), linewidth=1.1, alpha=0.85,
            )
        )
    fig.add_artist(
        Line2D(
            [reaching_box.x0 - 0.01, trial_box.x1 + 0.01], [y_middle, y_middle],
            transform=fig.transFigure, color=separator_color,
            linestyle=(0, (6, 4)), linewidth=1.1, alpha=0.85,
        )
    )
    fig.add_artist(
        Rectangle(
            (0.012, 0.018), 0.976, 0.964, transform=fig.transFigure,
            fill=False, edgecolor="black", linewidth=1.0,
        )
    )


def write_manifest(output_dir: Path, provenance: dict[str, Any], representative_trial: int) -> None:
    manifest = {
        "artifact": COMPOSITE_STEM,
        "source_analysis_contract": provenance["analysis_contract"],
        "num_paired_tasks": provenance["num_trials"],
        "common_success_tasks": provenance["common_success_trials"],
        "representative_trial": representative_trial,
        "representative_trial_reason": (
            "fixed shared representative task retained for cross-configuration visual comparison"
        ),
        "display_method_names": MAGAZINE_DISPLAY_NAMES,
        "high_outage_threshold": provenance["high_outage_threshold"],
        "radiomap_sha256": provenance["radiomap_sha256"],
        "trial_metrics_sha256": provenance["trial_metrics_sha256"],
        "panels": {
            "a": "common radio map, 100 paired starts, fixed goal, and high-outage contour",
            "b": "all-task cumulative goal-reaching rate versus step budget",
            "c": "all-trajectory mean cumulative outage versus executed step",
            "d": "three-method spatial trajectories for the representative trial",
            "e": "three-method common-map outage traces for the same representative trial",
        },
    }
    (output_dir / "case_study_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    provenance = read_provenance(analysis_dir)
    radio_map_path, directories = validate_inputs(provenance)
    targets = output_paths(output_dir)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing magazine figures: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    radio_map = RadioMap.from_npz(radio_map_path)
    threshold = float(provenance["high_outage_threshold"])
    tables: dict[str, list[dict[str, Any]]] = {}
    rollouts: dict[str, list[dict[str, np.ndarray]]] = {}
    for method in METHODS:
        tables[method], rollouts[method], _ = load_method(
            method, directories[method], radio_map, threshold
        )
    validate_pairing(tables)
    trial_count = len(tables[METHODS[0]])
    if trial_count != int(provenance["num_trials"]):
        raise ValueError("trial count does not match analysis provenance")
    trial = int(args.representative_trial)
    if trial < 0 or trial >= trial_count:
        raise ValueError(f"representative trial must be in [0, {trial_count - 1}]")
    if not all(tables[method][trial]["success"] for method in METHODS):
        raise ValueError("representative trial must be successful for all three methods")
    max_steps = max(int(row["steps"]) for method in METHODS for row in tables[method])

    with plt.rc_context(STANDALONE_STYLE):
        standalone_panel(
            output_dir, PANEL_STEMS["environment"], (7.0, 6.8),
            lambda axis, fig: plot_environment(axis, fig, radio_map, tables, threshold, title=True),
        )
        standalone_panel(
            output_dir, PANEL_STEMS["reaching"], (7.0, 4.8),
            lambda axis, fig: plot_reaching(axis, tables, max_steps, title=False),
        )
        standalone_panel(
            output_dir, PANEL_STEMS["cumulative_outage"], (7.0, 4.8),
            lambda axis, fig: plot_cumulative_outage(axis, rollouts, max_steps, title=False),
        )
        standalone_panel(
            output_dir, PANEL_STEMS["trajectory"], (7.0, 6.5),
            lambda axis, fig: plot_representative_trajectory(axis, radio_map, rollouts, trial, title=False),
        )
        standalone_panel(
            output_dir, PANEL_STEMS["trial_outage"], (7.0, 4.8),
            lambda axis, fig: plot_representative_outage(axis, rollouts, trial, threshold, title=True),
        )

    with plt.rc_context(COMPOSITE_STYLE):
        fig = plt.figure(figsize=(16.0, 8.8))
        grid = fig.add_gridspec(2, 3, width_ratios=(1.60, 1.0, 1.0))
        environment_axis = fig.add_subplot(grid[:, 0])
        reaching_axis = fig.add_subplot(grid[0, 1])
        cumulative_axis = fig.add_subplot(grid[1, 1])
        trajectory_axis = fig.add_subplot(grid[0, 2])
        trial_outage_axis = fig.add_subplot(grid[1, 2])
        plot_environment(environment_axis, fig, radio_map, tables, threshold, title=True, compact=True)
        plot_reaching(reaching_axis, tables, max_steps, title=True, compact=True)
        plot_cumulative_outage(cumulative_axis, rollouts, max_steps, title=True, compact=True)
        plot_representative_trajectory(
            trajectory_axis, radio_map, rollouts, trial, title=True, compact=True
        )
        plot_representative_outage(
            trial_outage_axis, rollouts, trial, threshold, title=True, compact=True
        )
        fig.subplots_adjust(left=0.045, right=0.985, top=0.935, bottom=0.105, wspace=0.27, hspace=0.42)
        add_magazine_separators(
            fig,
            (environment_axis, reaching_axis, cumulative_axis, trajectory_axis, trial_outage_axis),
        )
        save_figure(fig, output_dir, COMPOSITE_STEM, composite=True)

    write_manifest(output_dir, provenance, trial)
    print(f"saved WWM-NAV magazine case study: {output_dir}")


if __name__ == "__main__":
    main()
