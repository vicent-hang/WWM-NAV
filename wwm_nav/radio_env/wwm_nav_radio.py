from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


DEFAULT_BS_LOC_KM = (
    (1.0, 1.0, 0.025),
    (1.5774, 1.333, 0.025),
    (1.0, 1.6667, 0.025),
    (0.4226, 1.3333, 0.025),
    (0.4226, 0.6667, 0.025),
    (1.0, 0.3333, 0.025),
    (1.5774, 0.6667, 0.025),
)

DEFAULT_SECTOR_BORE_SITE_DEG = (-120.0, 0.0, 120.0)


@dataclass(frozen=True)
class SnarmRadioConfig:
    alpha: float = 0.3
    beta: int = 300
    gamma: float = 50.0
    max_height_m: float = 90.0
    sir_threshold_db: float = 0.0
    area_km: int = 2
    cluster_per_side: int = 3
    road_width_km: float = 0.02
    step: int = 101
    uav_height_km: float = 0.1
    bs_downtilt_deg: float = 100.0
    bs_power_watt: float = 0.1
    bs_loc_km: tuple[tuple[float, float, float], ...] | None = None
    sector_boresite_deg: tuple[float, ...] = DEFAULT_SECTOR_BORE_SITE_DEG
    carrier_freq_ghz: float = 2.0
    fast_fading_samples: int = 1000
    los_samples: int = 100


@dataclass(frozen=True)
class SnarmRadioMapData:
    outage: np.ndarray
    x_vec: np.ndarray
    y_vec: np.ndarray
    test_loc_m: np.ndarray
    test_outage: np.ndarray
    best_bs: np.ndarray
    best_sector: np.ndarray
    height_map: np.ndarray
    bs_loc_km: np.ndarray
    sector_boresite_deg: np.ndarray
    bs_power_watt: float
    uav_height_km: float
    max_height_m: float


class SnarmLikeRadioEnvironment:
    """SNARM-style radio environment without import-time side effects.

    This ports the core logic from SNARM-UAV-Learning/radio_environment.py:
    building generation, LoS/NLoS check, antenna gain, 3GPP path loss,
    Rician/Rayleigh fast fading, and empirical outage probability.
    """

    def __init__(self, config: SnarmRadioConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SnarmRadioConfig()
        self.rng = np.random.RandomState(seed)
        self.random = random.Random(seed)
        self.light_speed = 3.0e8
        self.wavelength_m = self.light_speed / (self.config.carrier_freq_ghz * 1.0e9)
        bs_loc_km = DEFAULT_BS_LOC_KM if self.config.bs_loc_km is None else self.config.bs_loc_km
        self.bs_loc_km = np.asarray(bs_loc_km, dtype=np.float64)
        if self.bs_loc_km.ndim != 2 or self.bs_loc_km.shape[1] != 3:
            raise ValueError(f"expected bs_loc_km shape (N, 3), got {self.bs_loc_km.shape}")
        if self.bs_loc_km.shape[0] < 1:
            raise ValueError("at least one base station is required")
        self.sector_boresite_deg = np.asarray(self.config.sector_boresite_deg, dtype=np.float64)
        if self.sector_boresite_deg.ndim != 1 or self.sector_boresite_deg.size < 1:
            raise ValueError("at least one sector boresite angle is required")
        self._build_city()

    @property
    def side_km(self) -> float:
        total_buildings = self.config.beta * (self.config.area_km**2)
        expected_area = self.config.alpha * (self.config.area_km**2) / total_buildings
        return float(np.sqrt(expected_area))

    @property
    def total_buildings(self) -> int:
        return int(self.config.beta * (self.config.area_km**2))

    def generate_radio_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = self.generate_radio_map_data()
        return data.outage, data.x_vec, data.y_vec, data.test_loc_m, data.test_outage

    def generate_radio_map_data(self) -> SnarmRadioMapData:
        x_vec = np.arange(self.config.area_km * (self.config.step - 1) + 1, dtype=np.int64)
        y_vec = np.arange(self.config.area_km * (self.config.step - 1) + 1, dtype=np.int64)
        outage_map = np.zeros((x_vec.size, y_vec.size), dtype=np.float64)
        best_bs_map = np.zeros((x_vec.size, y_vec.size), dtype=np.int64)
        best_sector_map = np.zeros((x_vec.size, y_vec.size), dtype=np.int64)
        loc_all = np.zeros((x_vec.size * y_vec.size, 3), dtype=np.float64)

        for i, x_idx in enumerate(x_vec):
            loc_vec = np.zeros((y_vec.size, 3), dtype=np.float64)
            loc_vec[:, 0] = x_idx / self.config.step
            loc_vec[:, 1] = y_vec / self.config.step
            loc_vec[:, 2] = self.config.uav_height_km
            loc_all[i * y_vec.size : (i + 1) * y_vec.size] = loc_vec
            for j, point_loc in enumerate(loc_vec):
                outage_matrix = self.get_point_outage_matrix(point_loc)
                flat_index = int(np.argmin(outage_matrix))
                bs_idx, sector_idx = np.unravel_index(flat_index, outage_matrix.shape)
                outage_map[i, j] = float(outage_matrix[bs_idx, sector_idx])
                best_bs_map[i, j] = int(bs_idx)
                best_sector_map[i, j] = int(sector_idx)

        outage_flat = outage_map.reshape(-1)
        test_size = int(loc_all.shape[0] / 10)
        test_indices = self.random.sample(range(loc_all.shape[0]), test_size)
        test_loc_m = loc_all[test_indices, :2] * 1000.0
        test_outage = outage_flat[test_indices]
        return SnarmRadioMapData(
            outage=outage_map,
            x_vec=x_vec,
            y_vec=y_vec,
            test_loc_m=test_loc_m,
            test_outage=test_outage,
            best_bs=best_bs_map,
            best_sector=best_sector_map,
            height_map=np.asarray(self.height_map, dtype=np.float64),
            bs_loc_km=np.asarray(self.bs_loc_km, dtype=np.float64),
            sector_boresite_deg=np.asarray(self.sector_boresite_deg, dtype=np.float64),
            bs_power_watt=float(self.config.bs_power_watt),
            uav_height_km=float(self.config.uav_height_km),
            max_height_m=float(self.config.max_height_m),
        )

    def get_point_min_outage(self, loc_vec_km: np.ndarray) -> np.ndarray:
        outages = np.zeros(loc_vec_km.shape[0], dtype=np.float64)
        for idx, point_loc in enumerate(loc_vec_km):
            outage_matrix = self.get_point_outage_matrix(point_loc)
            outages[idx] = np.min(outage_matrix)
        return outages

    def get_point_outage_matrix(self, point_loc_km: np.ndarray) -> np.ndarray:
        signal_from_bs = []
        total_power = 0.0
        for bs in self.bs_loc_km:
            los = self.check_los(point_loc_km, bs)
            measured_signal = self.get_received_power(point_loc_km, bs, los)
            signal_from_bs.append(measured_signal)
            total_power = total_power + measured_signal

        total_power_all_sector = np.sum(total_power, axis=1)
        outage_matrix = np.zeros((len(self.bs_loc_km), len(self.sector_boresite_deg)), dtype=np.float64)
        for bs_idx, signal_from_this_bs in enumerate(signal_from_bs):
            for sector in range(len(self.sector_boresite_deg)):
                signal_from_this_sector = signal_from_this_bs[:, sector]
                sir = signal_from_this_sector / (total_power_all_sector - signal_from_this_sector)
                sir_db = 10.0 * np.log10(sir)
                outage_matrix[bs_idx, sector] = np.sum(sir_db < self.config.sir_threshold_db) / len(sir_db)
        return outage_matrix

    def get_received_power(self, point_loc_km: np.ndarray, bs_km: np.ndarray, los: bool) -> np.ndarray:
        horizontal_distance = np.sqrt((bs_km[0] - point_loc_km[0]) ** 2 + (bs_km[1] - point_loc_km[1]) ** 2)
        theta = np.arctan((bs_km[2] - point_loc_km[2]) / horizontal_distance)
        theta_deg = np.rad2deg(theta) + 90.0
        if (point_loc_km[1] == bs_km[1]) and (point_loc_km[0] == bs_km[0]):
            phi = 0.0
        else:
            phi = np.arctan((point_loc_km[1] - bs_km[1]) / (point_loc_km[0] - bs_km[0] + 1e-5))
        phi_deg = np.rad2deg(phi)
        if (point_loc_km[1] > bs_km[1]) and (point_loc_km[0] < bs_km[0]):
            phi_deg += 180.0
        elif (point_loc_km[1] < bs_km[1]) and (point_loc_km[0] < bs_km[0]):
            phi_deg -= 180.0

        large_scale = self.get_large_scale_power(point_loc_km, bs_km, theta_deg, phi_deg, los)
        rayleigh = np.sqrt(0.5) * (
            self.rng.randn(self.config.fast_fading_samples, 3)
            + 1j * self.rng.randn(self.config.fast_fading_samples, 3)
        )

        if los:
            k_r_db = 15.0
            k_r = 10.0 ** (k_r_db / 10.0)
            distance_m = 1000.0 * np.sqrt(
                (bs_km[0] - point_loc_km[0]) ** 2
                + (bs_km[1] - point_loc_km[1]) ** 2
                + (bs_km[2] - point_loc_km[2]) ** 2
            )
            deterministic = np.exp(-1j * 2.0 * np.pi * distance_m / self.wavelength_m)
            fast_fading = np.sqrt(k_r / (k_r + 1.0)) * deterministic + np.sqrt(1.0 / (k_r + 1.0)) * rayleigh
        else:
            fast_fading = rayleigh

        h_overall = fast_fading * np.sqrt(np.tile(large_scale, (self.config.fast_fading_samples, 1)))
        return np.abs(h_overall) ** 2

    def check_los(self, point_loc_km: np.ndarray, bs_km: np.ndarray) -> bool:
        sample_points = np.linspace(0.0, 1.0, self.config.los_samples)
        x_sample = bs_km[0] + sample_points * (point_loc_km[0] - bs_km[0])
        y_sample = bs_km[1] + sample_points * (point_loc_km[1] - bs_km[1])
        z_sample = bs_km[2] + sample_points * (point_loc_km[2] - bs_km[2])
        x_range = np.floor(x_sample * (self.config.step - 1))
        y_range = np.floor(y_sample * (self.config.step - 1))
        x_range = np.maximum(x_range, 0)
        y_range = np.maximum(y_range, 0)
        idx_vec = np.int_(x_range * self.config.area_km * self.config.step + y_range)
        selected_height = np.asarray([self.height_map_array[0, i] for i in idx_vec])
        return not np.any(selected_height > z_sample)

    def get_large_scale_power(
        self,
        point_loc_km: np.ndarray,
        bs_km: np.ndarray,
        theta_deg: float,
        phi_deg: float,
        los: bool,
    ) -> np.ndarray:
        phi_sector_ref = phi_deg - self.sector_boresite_deg.copy()
        phi_sector_ref[phi_sector_ref < -180.0] += 360.0
        phi_sector_ref[phi_sector_ref > 180.0] -= 360.0
        channel_gain_db = np.zeros((1, len(self.sector_boresite_deg)), dtype=np.float64)
        for sector in range(len(self.sector_boresite_deg)):
            channel_gain_db[0, sector], _ = self.get_antenna_gain(theta_deg, phi_sector_ref[sector])
        channel_gain = np.power(10.0, channel_gain_db / 10.0)
        distance_m = 1000.0 * np.sqrt(
            (bs_km[0] - point_loc_km[0]) ** 2
            + (bs_km[1] - point_loc_km[1]) ** 2
            + (bs_km[2] - point_loc_km[2]) ** 2
        )

        if los:
            path_loss_db = 28.0 + 22.0 * np.log10(distance_m) + 20.0 * np.log10(self.config.carrier_freq_ghz)
        else:
            path_loss_db = (
                -17.5
                + (46.0 - 7.0 * np.log10(point_loc_km[2] * 1000.0)) * np.log10(distance_m)
                + 20.0 * np.log10(40.0 * np.pi * self.config.carrier_freq_ghz / 3.0)
            )
        path_loss_linear = 10.0 ** (-path_loss_db / 10.0)
        return channel_gain * self.config.bs_power_watt * path_loss_linear

    def get_antenna_gain(self, theta_deg: float, phi_deg: float) -> tuple[float, np.ndarray]:
        element_horizontal = 1
        element_vertical = 8
        dv = 0.5 * self.wavelength_m
        dh = 0.5 * self.wavelength_m
        angle_tilt_v = self.config.bs_downtilt_deg
        angle_tilt_h = 0.0
        angle_3db = 65.0
        am = 30.0
        ah = -np.min([12.0 * (phi_deg / angle_3db) ** 2, am])
        av = -np.min([12.0 * ((theta_deg - 90.0) / angle_3db) ** 2, am])
        ge_max = 8.0
        a_element = -np.min([-(ah + av), am])
        gelement_db = ge_max + a_element
        f_element = np.sqrt(10.0 ** (gelement_db / 10.0))

        k = 2.0 * np.pi / self.wavelength_m
        k_vector = k * np.array(
            [
                np.sin(theta_deg / 180.0 * np.pi) * np.cos(phi_deg / 180.0 * np.pi),
                np.sin(theta_deg / 180.0 * np.pi) * np.sin(phi_deg / 180.0 * np.pi),
                np.cos(theta_deg / 180.0 * np.pi),
            ],
            dtype=np.float64,
        )
        r_matrix = np.zeros((element_horizontal * element_vertical, 3), dtype=np.float64)
        for n in range(element_horizontal):
            r_matrix[(n + 1) * np.arange(element_vertical), 2] = np.arange(element_vertical) * dv
            r_matrix[(n + 1) * np.arange(element_vertical), 1] = n * dh
        steering_vector = np.exp(-1j * r_matrix.dot(k_vector.T))
        weight_vertical = (1.0 / np.sqrt(element_vertical)) * np.exp(
            -1j * k * np.arange(element_vertical) * dv * np.cos(angle_tilt_v / 180.0 * np.pi)
        )
        weight_horizontal = (1.0 / np.sqrt(element_horizontal)) * np.exp(
            -1j * k * np.arange(element_horizontal) * dh * np.sin(angle_tilt_h / 180.0 * np.pi)
        )
        weight_2d = np.kron(weight_horizontal, np.transpose(weight_vertical))
        weight_flatten = weight_2d.reshape(1, element_vertical * element_horizontal)
        array_factor = np.conjugate(weight_flatten).dot(steering_vector.reshape(element_vertical, 1))
        f_array = f_element * array_factor
        g_array = np.abs(f_array) ** 2
        return float(10.0 * np.log10(g_array).squeeze()), f_array

    def _build_city(self) -> None:
        cfg = self.config
        total_buildings = self.total_buildings
        height_vec = self.rng.rayleigh(cfg.gamma, total_buildings)
        height_vec = np.minimum(height_vec, cfg.max_height_m)

        cluster_count = cfg.cluster_per_side**2
        buildings_per_cluster = [np.ceil(total_buildings / cluster_count) for _ in range(cluster_count)]
        extra_building = int(np.sum(buildings_per_cluster) - total_buildings)
        if extra_building > 1:
            buildings_per_cluster[: extra_building - 1] = [
                np.ceil(total_buildings / cluster_count) - 1 for _ in range(extra_building)
            ]

        cluster_size = (cfg.area_km - (cfg.cluster_per_side - 1) * cfg.road_width_km) / cfg.cluster_per_side
        cluster_center = np.arange(cfg.cluster_per_side) * (cluster_size + cfg.road_width_km) + cluster_size / 2.0
        xloc: list[float] = []
        yloc: list[float] = []
        side = self.side_km
        for i in range(cfg.cluster_per_side):
            for j in range(cfg.cluster_per_side):
                idx = i * cfg.cluster_per_side + j
                buildings = int(buildings_per_cluster[idx])
                center_loc = [cluster_center[i], cluster_center[j]]
                buildings_per_row = int(np.ceil(np.sqrt(buildings)))
                x_loc = np.linspace(
                    (-cluster_size + 2.0 * side) / 2.0,
                    (cluster_size - 2.0 * side) / 2.0,
                    buildings_per_row,
                )
                loc_temp_x = np.array(list(x_loc) * buildings_per_row) + center_loc[0]
                loc_temp_y = np.repeat(list(x_loc), buildings_per_row) + center_loc[1]
                xloc.extend(list(loc_temp_x[:buildings]))
                yloc.extend(list(loc_temp_y[:buildings]))

        height_map = np.zeros((cfg.area_km * cfg.step, cfg.area_km * cfg.step), dtype=np.float64)
        for i in range(total_buildings):
            x1 = xloc[i] - side / 2.0
            x2 = xloc[i] + side / 2.0
            y1 = yloc[i] - side / 2.0
            y2 = yloc[i] + side / 2.0
            height_map[int(np.ceil(x1 * cfg.step) - 1) : int(np.floor(x2 * cfg.step)),
                       int(np.ceil(y1 * cfg.step) - 1) : int(np.floor(y2 * cfg.step))] = height_vec[i]

        self.xloc = np.asarray(xloc, dtype=np.float64)
        self.yloc = np.asarray(yloc, dtype=np.float64)
        self.height_map = height_map
        self.height_map_array = height_map.reshape(1, (cfg.area_km * cfg.step) ** 2)


def generate_snarm_like_radio_env(
    seed: int | None = None,
    fast_fading_samples: int = 1000,
    los_samples: int = 100,
    bs_loc_km: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    sector_boresite_deg: tuple[float, ...] | np.ndarray | None = None,
    bs_power_watt: float = 0.1,
    return_physics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | SnarmRadioMapData:
    config = SnarmRadioConfig(
        fast_fading_samples=fast_fading_samples,
        los_samples=los_samples,
        bs_loc_km=tuple(map(tuple, np.asarray(bs_loc_km, dtype=np.float64))) if bs_loc_km is not None else None,
        sector_boresite_deg=tuple(np.asarray(sector_boresite_deg, dtype=np.float64))
        if sector_boresite_deg is not None
        else DEFAULT_SECTOR_BORE_SITE_DEG,
        bs_power_watt=float(bs_power_watt),
    )
    env = SnarmLikeRadioEnvironment(config=config, seed=seed)
    if return_physics:
        return env.generate_radio_map_data()
    return env.generate_radio_map()
