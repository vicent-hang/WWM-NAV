from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wwm-nav")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage
from matplotlib.patches import Ellipse, Patch
from mpl_toolkits.mplot3d import proj3d

from wwm_nav.radio_env.wwm_nav_radio import (
    DEFAULT_BS_LOC_KM,
    DEFAULT_SECTOR_BORE_SITE_DEG,
    SnarmLikeRadioEnvironment,
    SnarmRadioConfig,
)
from wwm_nav.radiomap import RadioMap
from wwm_nav.tools.magazine_data import (
    read_provenance,
    validate_inputs,
)
from wwm_nav.tools.plotting import configure_matplotlib

OUTPUT_STEM = "panel_city_bs_radio_coverage_25d"
STYLE = {
    "font.size": 13.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 18.0,
    "figure.dpi": 180,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
TEXT_SCALE = 1.4


def scale_descriptive_text(fig: Any, factor: float) -> None:
    for axis in fig.axes:
        labels = [axis.title, axis.xaxis.label, axis.yaxis.label]
        if hasattr(axis, "zaxis"):
            labels.append(axis.zaxis.label)
        for text in labels:
            text.set_fontsize(text.get_fontsize() * factor)
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontweight("bold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a reproducible 2.5D city, base-station layout, and outage "
            "coverage view for the current WWM-NAV radio environment."
        )
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("reference/magazine/data")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs")
    )
    parser.add_argument("--city-seed", type=int, default=7)
    parser.add_argument("--elevation", type=float, default=40.0)
    parser.add_argument("--azimuth", type=float, default=-52.0)
    parser.add_argument(
        "--uav-icon",
        type=Path,
        default=Path("reference/magazine/icon/uav-dajiang.png"),
    )
    parser.add_argument("--uav-position", nargs=2, type=float, default=(1270.0, 830.0))
    parser.add_argument("--uav-size-m", type=float, default=92.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / f"{OUTPUT_STEM}.png",
        output_dir / f"{OUTPUT_STEM}.pdf",
        output_dir / f"{OUTPUT_STEM}.json",
    ]


def building_geometry(
    city: SnarmLikeRadioEnvironment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = city.config
    side_m = float(city.side_km * 1000.0)
    x_m = np.asarray(city.xloc, dtype=np.float64) * 1000.0
    y_m = np.asarray(city.yloc, dtype=np.float64) * 1000.0
    heights = np.zeros(x_m.size, dtype=np.float64)
    for index, (x_km, y_km) in enumerate(zip(city.xloc, city.yloc, strict=True)):
        x1 = float(x_km - city.side_km / 2.0)
        x2 = float(x_km + city.side_km / 2.0)
        y1 = float(y_km - city.side_km / 2.0)
        y2 = float(y_km + city.side_km / 2.0)
        ix1 = max(0, int(np.ceil(x1 * config.step) - 1))
        ix2 = min(city.height_map.shape[0], int(np.floor(x2 * config.step)))
        iy1 = max(0, int(np.ceil(y1 * config.step) - 1))
        iy2 = min(city.height_map.shape[1], int(np.floor(y2 * config.step)))
        patch = city.height_map[ix1:ix2, iy1:iy2]
        heights[index] = float(np.max(patch)) if patch.size else 0.0
    keep = heights > 0.0
    return x_m[keep], y_m[keep], heights[keep], np.full(np.count_nonzero(keep), side_m)


def draw_base_stations(
    axis: Any,
    bs_xyz_m: np.ndarray,
    coverage_height_m: float,
) -> None:
    sector_length_m = 62.0
    for index, (x_m, y_m, height_m) in enumerate(bs_xyz_m, start=1):
        display_height_m = coverage_height_m + 8.0
        axis.plot(
            [x_m, x_m],
            [y_m, y_m],
            [0.0, height_m],
            color="#B2182B",
            linewidth=2.6,
            zorder=12,
        )
        axis.plot(
            [x_m, x_m],
            [y_m, y_m],
            [height_m, coverage_height_m],
            color="#B2182B",
            linestyle="--",
            linewidth=1.0,
            alpha=0.82,
            zorder=12,
        )
        axis.plot(
            [x_m, x_m],
            [y_m, y_m],
            [coverage_height_m, display_height_m],
            color="#B2182B",
            linewidth=2.2,
            zorder=13,
        )
        for angle_deg in DEFAULT_SECTOR_BORE_SITE_DEG:
            angle_rad = np.deg2rad(angle_deg)
            axis.plot(
                [x_m, x_m + sector_length_m * np.cos(angle_rad)],
                [y_m, y_m + sector_length_m * np.sin(angle_rad)],
                [display_height_m, display_height_m],
                color="#B2182B",
                linewidth=1.4,
                zorder=12,
            )
        axis.scatter(
            [x_m],
            [y_m],
            [display_height_m],
            marker="^",
            s=42,
            color="#D73027",
            edgecolor="white",
            linewidth=0.8,
            depthshade=False,
            zorder=13,
        )
        axis.text(
            x_m,
            y_m,
            display_height_m + 4.0,
            f"BS{index}",
            color="#7F0000",
            fontsize=7.5,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.64,
            },
            zorder=14,
        )


def draw_uav_icon(
    axis: Any,
    icon_path: Path,
    position_xy_m: tuple[float, float],
    size_m: float,
    height_m: float,
) -> None:
    icon = np.asarray(mpimg.imread(icon_path), dtype=np.float64)
    if icon.ndim != 3 or icon.shape[2] != 4:
        raise ValueError(f"UAV icon must be an RGBA image: {icon_path}")
    projected_x, projected_y, _ = proj3d.proj_transform(
        float(position_xy_m[0]),
        float(position_xy_m[1]),
        float(height_m),
        axis.get_proj(),
    )
    shadow_x, shadow_y, _ = proj3d.proj_transform(
        float(position_xy_m[0]),
        float(position_xy_m[1]),
        float(height_m - 3.2),
        axis.get_proj(),
    )
    shadow = DrawingArea(34.0, 13.0, 0.0, 0.0)
    shadow.add_artist(
        Ellipse(
            (17.0, 6.5),
            width=29.0,
            height=7.0,
            facecolor="#111111",
            edgecolor="none",
            alpha=0.24,
        )
    )
    axis.add_artist(
        AnnotationBbox(
            shadow,
            (shadow_x, shadow_y),
            xycoords="data",
            frameon=False,
            pad=0.0,
            annotation_clip=False,
            zorder=99,
        )
    )
    artist = AnnotationBbox(
        OffsetImage(icon, zoom=max(0.12, float(size_m) / 480.0)),
        (projected_x, projected_y),
        xycoords="data",
        frameon=False,
        pad=0.0,
        annotation_clip=False,
        zorder=100,
    )
    axis.add_artist(artist)


def plot_city_radio_map(
    radio_map: RadioMap,
    city: SnarmLikeRadioEnvironment,
    threshold: float,
    elevation: float,
    azimuth: float,
    uav_icon_path: Path | None,
    uav_position_xy_m: tuple[float, float],
    uav_size_m: float,
) -> Any:
    fig = plt.figure(figsize=(8.6, 7.0))
    axis = fig.add_axes([0.005, 0.020, 0.99, 0.975], projection="3d")
    x_mesh, y_mesh = np.meshgrid(radio_map.x_m, radio_map.y_m, indexing="xy")
    outage = np.asarray(radio_map.outage.T, dtype=np.float64)
    norm = Normalize(vmin=0.0, vmax=1.0)
    coverage_cmap = colormaps["viridis"]

    coverage_height_m = float(city.config.uav_height_km * 1000.0)
    ground_z = np.full_like(outage, 0.8)
    axis.plot_surface(
        x_mesh,
        y_mesh,
        ground_z,
        rstride=2,
        cstride=2,
        facecolors=coverage_cmap(norm(outage)),
        shade=False,
        antialiased=False,
        linewidth=0.0,
        alpha=0.36,
        zorder=0,
    )
    coverage_z = np.full_like(outage, coverage_height_m)
    axis.plot_surface(
        x_mesh,
        y_mesh,
        coverage_z,
        rstride=2,
        cstride=2,
        facecolors=coverage_cmap(norm(outage)),
        shade=False,
        antialiased=False,
        linewidth=0.0,
        alpha=0.56,
        zorder=10,
    )
    axis.contour(
        x_mesh,
        y_mesh,
        outage,
        levels=[threshold],
        zdir="z",
        offset=coverage_height_m + 0.8,
        colors="#202020",
        linestyles="--",
        linewidths=0.75,
        zorder=2,
    )

    building_x, building_y, building_h, building_side = building_geometry(city)
    height_norm = np.clip(building_h / max(float(city.config.max_height_m), 1.0), 0.0, 1.0)
    building_colors = colormaps["Greys"](0.20 + 0.43 * height_norm)
    building_colors[:, 3] = 0.72
    axis.bar3d(
        building_x - building_side / 2.0,
        building_y - building_side / 2.0,
        np.full_like(building_h, 2.0),
        building_side,
        building_side,
        building_h,
        color=building_colors,
        edgecolor="#50575E",
        linewidth=0.12,
        shade=True,
        zsort="average",
        zorder=5,
    )

    bs_xyz_m = np.asarray(DEFAULT_BS_LOC_KM, dtype=np.float64) * 1000.0
    draw_base_stations(axis, bs_xyz_m, coverage_height_m)

    axis.set_xlim(float(radio_map.x_m[0]), float(radio_map.x_m[-1]))
    axis.set_ylim(float(radio_map.y_m[0]), float(radio_map.y_m[-1]))
    axis.set_zlim(0.0, max(120.0, coverage_height_m + 15.0))
    axis.set_xlabel("x (m)", labelpad=7)
    axis.set_ylabel("y (m)", labelpad=7)
    axis.set_zlabel("Height (m)", labelpad=5)
    axis.set_xticks(np.arange(0.0, 2000.1, 400.0))
    axis.set_yticks(np.arange(0.0, 2000.1, 400.0))
    axis.set_zticks(np.arange(0.0, 101.0, 25.0))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_box_aspect((1.0, 1.0, 0.32))
    if uav_icon_path is not None:
        draw_uav_icon(
            axis,
            uav_icon_path,
            uav_position_xy_m,
            uav_size_m,
            coverage_height_m + 4.0,
        )
    axis.grid(True, linewidth=0.45, alpha=0.45)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.96, 0.97, 0.98, 0.86))

    axis.legend(
        handles=[
            Patch(facecolor="#AEB5BC", edgecolor="#50575E", label="Buildings"),
            Line2D(
                [0], [0], marker="^", linestyle="none", color="#D73027",
                markeredgecolor="white", markersize=8, label="Base stations",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        framealpha=0.92,
    )
    return fig


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    uav_icon_path = None if args.uav_icon is None else args.uav_icon.resolve()
    if uav_icon_path is not None and not uav_icon_path.is_file():
        raise FileNotFoundError(uav_icon_path)
    provenance = read_provenance(analysis_dir)
    radio_map_path, _ = validate_inputs(provenance)
    targets = output_paths(output_dir)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing 2.5D outputs: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    radio_map = RadioMap.from_npz(radio_map_path)
    city_config = SnarmRadioConfig()
    city = SnarmLikeRadioEnvironment(config=city_config, seed=int(args.city_seed))
    threshold = float(provenance["high_outage_threshold"])
    with plt.rc_context(STYLE):
        fig = plot_city_radio_map(
            radio_map,
            city,
            threshold,
            float(args.elevation),
            float(args.azimuth),
            uav_icon_path,
            (float(args.uav_position[0]), float(args.uav_position[1])),
            float(args.uav_size_m),
        )
        scale_descriptive_text(fig, TEXT_SCALE)
        fig.savefig(targets[0], dpi=320, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(targets[1], bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    manifest = {
        "artifact": OUTPUT_STEM,
        "analysis_contract": provenance["analysis_contract"],
        "radio_map_source": str(radio_map_path),
        "radio_map_sha256": provenance["radiomap_sha256"],
        "coverage_semantics": "stored outage probability from the WWM-NAV evaluation map",
        "coverage_plane_height_m": float(city_config.uav_height_km * 1000.0),
        "physics_metadata_boundary": (
            "legacy radio-map artifact stores arr_0 through arr_4 only; city and BS metadata "
            "are reconstructed from current repository defaults rather than read from the artifact"
        ),
        "city_source": "SnarmLikeRadioEnvironment(SnarmRadioConfig(), seed=city_seed)",
        "city_seed": int(args.city_seed),
        "city_parameters": {
            "alpha": city_config.alpha,
            "beta": city_config.beta,
            "gamma": city_config.gamma,
            "max_height_m": city_config.max_height_m,
            "building_count": int(city.total_buildings),
            "building_layout_contract": "snarm_regular_grid_v1",
        },
        "base_station_source": "radiomap_nav_wm.radio_env.snarm_like.DEFAULT_BS_LOC_KM",
        "base_station_locations_km": [list(values) for values in DEFAULT_BS_LOC_KM],
        "base_station_height_m": float(np.asarray(DEFAULT_BS_LOC_KM)[0, 2] * 1000.0),
        "sector_boresites_deg": list(DEFAULT_SECTOR_BORE_SITE_DEG),
        "high_outage_threshold": threshold,
        "view": {"elevation_deg": float(args.elevation), "azimuth_deg": float(args.azimuth)},
        "uav_icon": None if uav_icon_path is None else str(uav_icon_path),
        "uav_position_xy_m": [float(args.uav_position[0]), float(args.uav_position[1])],
        "uav_display_size_m": float(args.uav_size_m),
        "uav_display_height_m": float(city_config.uav_height_km * 1000.0 + 4.0),
        "uav_trajectory_drawn": False,
    }
    targets[2].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved 2.5D city/base-station radio-map view: {targets[0]}")


if __name__ == "__main__":
    main()
