from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RadioMap:
    """Precomputed outage-probability radio map.

    The SNARM reference file stores X/Y vectors as 0..200 and plots them after
    multiplying by 10, so the default coordinate scale converts map coordinates
    to meters.
    """

    outage: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    bs_loc_km: np.ndarray | None = None
    sector_boresite_deg: np.ndarray | None = None
    bs_power_watt: float | None = None
    height_map: np.ndarray | None = None
    best_bs: np.ndarray | None = None
    best_sector: np.ndarray | None = None
    uav_height_km: float = 0.1
    max_height_m: float = 90.0
    bs_downtilt_deg: float = 100.0
    carrier_freq_ghz: float = 2.0
    los_samples: int = 100

    @classmethod
    def from_npz(cls, path: str | Path, coordinate_scale: float = 10.0, require_physics: bool = False) -> "RadioMap":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"radio map file not found: {path}. "
                "Generate one with: python scripts/generate_radio_env.py"
            )
        data = np.load(path, allow_pickle=False)
        outage = np.asarray(data["arr_0"], dtype=np.float64)
        x_m = np.asarray(data["arr_1"], dtype=np.float64) * coordinate_scale
        y_m = np.asarray(data["arr_2"], dtype=np.float64) * coordinate_scale

        if outage.ndim != 2:
            raise ValueError(f"expected 2D outage map, got shape {outage.shape}")
        if outage.shape != (x_m.size, y_m.size):
            raise ValueError(
                "outage shape does not match x/y vectors: "
                f"{outage.shape} vs {(x_m.size, y_m.size)}"
            )
        if np.nanmin(outage) < 0.0 or np.nanmax(outage) > 1.0:
            raise ValueError("outage probabilities must be in [0, 1]")

        fields = set(data.files)
        physics_fields = {
            "bs_loc_km",
            "sector_boresite_deg",
            "bs_power_watt",
            "height_map",
            "best_bs",
            "best_sector",
        }
        if require_physics and not physics_fields.issubset(fields):
            missing = sorted(physics_fields.difference(fields))
            raise ValueError(
                "radio map is missing physics fields required by physics-conditioned RSSM+Q: "
                f"{missing}. Regenerate it with: python scripts/generate_radio_env.py"
            )

        bs_loc_km = np.asarray(data["bs_loc_km"], dtype=np.float64) if "bs_loc_km" in fields else None
        sector_boresite_deg = (
            np.asarray(data["sector_boresite_deg"], dtype=np.float64) if "sector_boresite_deg" in fields else None
        )
        height_map = np.asarray(data["height_map"], dtype=np.float64) if "height_map" in fields else None
        best_bs = np.asarray(data["best_bs"], dtype=np.int64) if "best_bs" in fields else None
        best_sector = np.asarray(data["best_sector"], dtype=np.int64) if "best_sector" in fields else None
        bs_power_watt = float(np.asarray(data["bs_power_watt"]).reshape(-1)[0]) if "bs_power_watt" in fields else None
        uav_height_km = (
            float(np.asarray(data["uav_height_km"]).reshape(-1)[0]) if "uav_height_km" in fields else 0.1
        )
        max_height_m = (
            float(np.asarray(data["max_height_m"]).reshape(-1)[0]) if "max_height_m" in fields else 90.0
        )
        bs_downtilt_deg = (
            float(np.asarray(data["bs_downtilt_deg"]).reshape(-1)[0])
            if "bs_downtilt_deg" in fields
            else 100.0
        )
        carrier_freq_ghz = (
            float(np.asarray(data["carrier_freq_ghz"]).reshape(-1)[0])
            if "carrier_freq_ghz" in fields
            else 2.0
        )
        los_samples = (
            int(np.asarray(data["los_samples"]).reshape(-1)[0])
            if "los_samples" in fields
            else 100
        )
        if best_bs is not None and best_bs.shape != outage.shape:
            raise ValueError(f"best_bs shape {best_bs.shape} does not match outage shape {outage.shape}")
        if best_sector is not None and best_sector.shape != outage.shape:
            raise ValueError(f"best_sector shape {best_sector.shape} does not match outage shape {outage.shape}")
        return cls(
            outage=outage,
            x_m=x_m,
            y_m=y_m,
            bs_loc_km=bs_loc_km,
            sector_boresite_deg=sector_boresite_deg,
            bs_power_watt=bs_power_watt,
            height_map=height_map,
            best_bs=best_bs,
            best_sector=best_sector,
            uav_height_km=uav_height_km,
            max_height_m=max_height_m,
            bs_downtilt_deg=bs_downtilt_deg,
            carrier_freq_ghz=carrier_freq_ghz,
            los_samples=los_samples,
        )

    @property
    def coverage(self) -> np.ndarray:
        return 1.0 - self.outage

    @property
    def resolution_m(self) -> tuple[float, float]:
        dx = float(np.median(np.diff(self.x_m)))
        dy = float(np.median(np.diff(self.y_m)))
        return dx, dy

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        return (
            float(self.x_m[0]),
            float(self.x_m[-1]),
            float(self.y_m[0]),
            float(self.y_m[-1]),
        )

    def nearest_index(self, xy_m: np.ndarray | list[float] | tuple[float, float]) -> tuple[int, int]:
        xy = np.asarray(xy_m, dtype=np.float64)
        if xy.shape != (2,):
            raise ValueError(f"expected xy shape (2,), got {xy.shape}")
        ix = int(np.abs(self.x_m - xy[0]).argmin())
        iy = int(np.abs(self.y_m - xy[1]).argmin())
        return ix, iy

    def index_to_xy(self, index: tuple[int, int]) -> np.ndarray:
        ix, iy = index
        return np.array([self.x_m[ix], self.y_m[iy]], dtype=np.float64)

    def outage_at_index(self, index: tuple[int, int]) -> float:
        return float(self.outage[index])

    def outage_at_xy(self, xy_m: np.ndarray | list[float] | tuple[float, float]) -> float:
        return self.outage_at_index(self.nearest_index(xy_m))

    @property
    def has_physics(self) -> bool:
        return (
            self.bs_loc_km is not None
            and self.sector_boresite_deg is not None
            and self.bs_power_watt is not None
            and self.height_map is not None
            and self.best_bs is not None
            and self.best_sector is not None
        )

    @property
    def sector_count(self) -> int:
        self.require_physics()
        assert self.sector_boresite_deg is not None
        return int(self.sector_boresite_deg.size)

    def require_physics(self) -> None:
        if not self.has_physics:
            raise ValueError(
                "this RadioMap has no physical BS/building fields; regenerate the map with "
                "python scripts/generate_radio_env.py"
            )

    @staticmethod
    def physical_context_dim(num_bs: int) -> int:
        return int(num_bs) * 11

    def building_patch_at_xy(
        self,
        xy_m: np.ndarray | list[float] | tuple[float, float],
        patch_size: int,
    ) -> np.ndarray:
        self.require_physics()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        assert self.height_map is not None
        ix, iy = self._height_map_index(xy_m)
        radius = patch_size // 2
        padded = np.pad(self.height_map, ((radius, radius), (radius, radius)), mode="edge")
        cx = ix + radius
        cy = iy + radius
        patch = padded[cx - radius : cx + radius + 1, cy - radius : cy + radius + 1]
        patch = np.asarray(patch, dtype=np.float32) / max(float(self.max_height_m), 1.0)
        return np.clip(patch, 0.0, 1.0)[None, :, :]

    def physical_context_at_xy(
        self,
        xy_m: np.ndarray | list[float] | tuple[float, float],
        num_bs: int,
    ) -> np.ndarray:
        self.require_physics()
        assert self.bs_loc_km is not None
        xy = np.asarray(xy_m, dtype=np.float64)
        if xy.shape != (2,):
            raise ValueError(f"expected xy shape (2,), got {xy.shape}")
        bs_m = self.bs_loc_km * 1000.0
        area_scale = max(float(max(self.x_m[-1] - self.x_m[0], self.y_m[-1] - self.y_m[0])), 1.0)
        diagonal = max(float(np.linalg.norm([self.x_m[-1] - self.x_m[0], self.y_m[-1] - self.y_m[0]])), 1.0)
        delta_xy = xy[None, :] - bs_m[:, :2]
        d2d = np.linalg.norm(delta_xy, axis=1)
        order = np.argsort(d2d)[: max(int(num_bs), 0)]
        rows = np.zeros((int(num_bs), 11), dtype=np.float32)
        uav_z_m = float(self.uav_height_km) * 1000.0
        power_norm = np.log10(max(float(self.bs_power_watt or 1.0e-12), 1.0e-12)) / 2.0
        for slot, bs_idx in enumerate(order):
            bs = bs_m[int(bs_idx)]
            dx = float(delta_xy[int(bs_idx), 0])
            dy = float(delta_xy[int(bs_idx), 1])
            dz = float(uav_z_m - bs[2])
            distance_2d = float(max(d2d[int(bs_idx)], 1.0e-6))
            distance_3d = float(np.sqrt(distance_2d**2 + dz**2))
            azimuth = float(np.arctan2(dy, dx))
            elevation = float(np.arctan2(dz, distance_2d))
            los, clearance_m = self.link_los_clearance(xy, self.bs_loc_km[int(bs_idx)])
            rows[slot] = np.asarray(
                [
                    dx / area_scale,
                    dy / area_scale,
                    distance_2d / diagonal,
                    distance_3d / diagonal,
                    elevation / np.pi,
                    np.sin(azimuth),
                    np.cos(azimuth),
                    power_norm,
                    1.0 if los else 0.0,
                    np.clip(clearance_m / max(float(self.max_height_m), 1.0), -1.0, 1.0),
                    1.0,
                ],
                dtype=np.float32,
            )
        return rows.reshape(-1)

    def link_los_clearance(
        self,
        xy_m: np.ndarray | list[float] | tuple[float, float],
        bs_km: np.ndarray,
        samples: int = 100,
    ) -> tuple[bool, float]:
        self.require_physics()
        assert self.height_map is not None
        point = np.asarray(xy_m, dtype=np.float64)
        point_km = np.asarray([point[0] / 1000.0, point[1] / 1000.0, float(self.uav_height_km)], dtype=np.float64)
        bs = np.asarray(bs_km, dtype=np.float64)
        t = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
        x_km = bs[0] + t * (point_km[0] - bs[0])
        y_km = bs[1] + t * (point_km[1] - bs[1])
        z_line = bs[2] + t * (point_km[2] - bs[2])
        ix, iy = self._height_map_indices_from_km(x_km, y_km)
        heights = self.height_map[ix, iy]
        clearance = z_line - heights
        min_clearance = float(np.min(clearance))
        return bool(np.all(clearance >= 0.0)), min_clearance

    def _height_map_index(self, xy_m: np.ndarray | list[float] | tuple[float, float]) -> tuple[int, int]:
        assert self.height_map is not None
        xy = np.asarray(xy_m, dtype=np.float64)
        x_min, x_max, y_min, y_max = self.bounds_m
        x_frac = (xy[0] - x_min) / max(x_max - x_min, 1.0)
        y_frac = (xy[1] - y_min) / max(y_max - y_min, 1.0)
        ix = int(np.clip(round(x_frac * (self.height_map.shape[0] - 1)), 0, self.height_map.shape[0] - 1))
        iy = int(np.clip(round(y_frac * (self.height_map.shape[1] - 1)), 0, self.height_map.shape[1] - 1))
        return ix, iy

    def _height_map_indices_from_km(self, x_km: np.ndarray, y_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.height_map is not None
        x_min, x_max, y_min, y_max = self.bounds_m
        x_m = np.asarray(x_km, dtype=np.float64) * 1000.0
        y_m = np.asarray(y_km, dtype=np.float64) * 1000.0
        x_frac = (x_m - x_min) / max(x_max - x_min, 1.0)
        y_frac = (y_m - y_min) / max(y_max - y_min, 1.0)
        ix = np.clip(np.floor(x_frac * (self.height_map.shape[0] - 1)).astype(np.int64), 0, self.height_map.shape[0] - 1)
        iy = np.clip(np.floor(y_frac * (self.height_map.shape[1] - 1)).astype(np.int64), 0, self.height_map.shape[1] - 1)
        return ix, iy
