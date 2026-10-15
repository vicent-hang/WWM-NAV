from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


C28_5_LEGACY_MAP_SHA256 = "f294973e9a6ffaded3802d54088665891b4e89d9c40e0c0b3eff0361c3fb278e"
C28_5_MEMBER_SHA256 = {
    "arr_0.npy": "6c88f29ddf6818b1cf36f44281b7d06257a4d0a20a3328c6d8e3d587193757fd",
    "arr_1.npy": "a937a22d8bc20dc95f2f00ea96bda7299899c719130b14fc1931c682dee17c5c",
    "arr_2.npy": "a937a22d8bc20dc95f2f00ea96bda7299899c719130b14fc1931c682dee17c5c",
    "arr_3.npy": "1e5fc00cd3f69cab0e869b7ccc83feccc7a2cc0f57224bec72a8c798e186e2a6",
    "arr_4.npy": "e07cf3022bb3e1715b746c4229836820fa766e7df2fa549106f5b7e556cbffca",
}
C28_5_MAP_CONTRACT = "c28_5_legacy_core_physics_reconstruction_v1"
C28_5_CITY_SEED = 7
C28_5_BS_LOC_KM = np.asarray(
    [
        (1.0, 1.0, 0.025),
        (1.5774, 1.333, 0.025),
        (1.0, 1.6667, 0.025),
        (0.4226, 1.3333, 0.025),
        (0.4226, 0.6667, 0.025),
        (1.0, 0.3333, 0.025),
        (1.5774, 0.6667, 0.025),
    ],
    dtype=np.float64,
)
C28_5_SECTOR_BORESITE_DEG = np.asarray((-120.0, 0.0, 120.0), dtype=np.float64)
C28_5_PREPARED_FIELDS = {
    "arr_0",
    "arr_1",
    "arr_2",
    "arr_3",
    "arr_4",
    "active_original_bs_ids",
    "best_bs",
    "best_link_labels_valid",
    "best_sector",
    "bs_downtilt_deg",
    "bs_loc_km",
    "bs_power_watt",
    "bs_power_watt_by_bs",
    "c28_5_legacy_map_sha256",
    "c28_5_map_contract",
    "carrier_freq_ghz",
    "height_map",
    "los_samples",
    "max_height_m",
    "sector_boresite_deg",
    "uav_height_km",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_member_sha256(path: Path, member: str) -> str:
    with ZipFile(path) as archive:
        return hashlib.sha256(archive.read(member)).hexdigest()


def semantic_npz_sha256(path: Path) -> str:
    """Hash NPZ array contents without depending on ZIP timestamps."""

    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = np.ascontiguousarray(np.asarray(data[name]))
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def reconstruct_c28_5_height_map(seed: int = C28_5_CITY_SEED) -> np.ndarray:
    """Recreate the SNARM-grid city used by the registered C28.5 map."""

    alpha = 0.3
    beta = 300
    gamma = 50.0
    max_height_m = 90.0
    area_km = 2
    cluster_per_side = 3
    road_width_km = 0.02
    step = 101
    total_buildings = beta * (area_km**2)
    rng = np.random.RandomState(int(seed))
    height_vec = np.minimum(rng.rayleigh(gamma, total_buildings), max_height_m)

    cluster_count = cluster_per_side**2
    buildings_per_cluster = [np.ceil(total_buildings / cluster_count) for _ in range(cluster_count)]
    extra_building = int(np.sum(buildings_per_cluster) - total_buildings)
    if extra_building > 1:
        buildings_per_cluster[: extra_building - 1] = [
            np.ceil(total_buildings / cluster_count) - 1 for _ in range(extra_building)
        ]

    cluster_size = (area_km - (cluster_per_side - 1) * road_width_km) / cluster_per_side
    cluster_center = (
        np.arange(cluster_per_side) * (cluster_size + road_width_km) + cluster_size / 2.0
    )
    side_km = float(np.sqrt(alpha * (area_km**2) / total_buildings))
    x_locations: list[float] = []
    y_locations: list[float] = []
    for x_index in range(cluster_per_side):
        for y_index in range(cluster_per_side):
            cluster_index = x_index * cluster_per_side + y_index
            building_count = int(buildings_per_cluster[cluster_index])
            buildings_per_row = int(np.ceil(np.sqrt(building_count)))
            row_locations = np.linspace(
                (-cluster_size + 2.0 * side_km) / 2.0,
                (cluster_size - 2.0 * side_km) / 2.0,
                buildings_per_row,
            )
            x_locations.extend(
                list(
                    (
                        np.asarray(list(row_locations) * buildings_per_row)
                        + cluster_center[x_index]
                    )[:building_count]
                )
            )
            y_locations.extend(
                list(
                    (
                        np.repeat(list(row_locations), buildings_per_row)
                        + cluster_center[y_index]
                    )[:building_count]
                )
            )

    height_map = np.zeros((area_km * step, area_km * step), dtype=np.float64)
    for building_index in range(total_buildings):
        x1 = x_locations[building_index] - side_km / 2.0
        x2 = x_locations[building_index] + side_km / 2.0
        y1 = y_locations[building_index] - side_km / 2.0
        y2 = y_locations[building_index] + side_km / 2.0
        height_map[
            int(np.ceil(x1 * step) - 1) : int(np.floor(x2 * step)),
            int(np.ceil(y1 * step) - 1) : int(np.floor(y2 * step)),
        ] = height_vec[building_index]
    return height_map


def verify_c28_5_legacy_map(path: Path) -> dict[str, str]:
    path = Path(path)
    actual_file_hash = file_sha256(path)
    if actual_file_hash != C28_5_LEGACY_MAP_SHA256:
        raise ValueError(
            "C28.5 legacy map SHA-256 mismatch: "
            f"expected {C28_5_LEGACY_MAP_SHA256}, got {actual_file_hash}"
        )
    actual_members = {
        member: npz_member_sha256(path, member) for member in C28_5_MEMBER_SHA256
    }
    mismatches = {
        member: (C28_5_MEMBER_SHA256[member], actual)
        for member, actual in actual_members.items()
        if actual != C28_5_MEMBER_SHA256[member]
    }
    if mismatches:
        raise ValueError(f"C28.5 legacy map member hash mismatch: {mismatches}")
    return actual_members


def prepare_c28_5_radiomap(
    legacy_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    legacy_path = Path(legacy_path)
    output_path = Path(output_path)
    member_hashes = verify_c28_5_legacy_map(legacy_path)
    with np.load(legacy_path, allow_pickle=False) as source:
        core = [np.asarray(source[f"arr_{index}"]) for index in range(5)]

    outage, x_axis, y_axis, test_locations, test_outages = core
    if outage.shape != (201, 201) or x_axis.shape != (201,) or y_axis.shape != (201,):
        raise ValueError("registered C28.5 map must use the 201x201 grid")
    height_map = reconstruct_c28_5_height_map()
    # C28.5 sets best_link_weight=0, so these compatibility labels are never
    # optimized or queried. They retain the historical loader schema only.
    best_bs = np.zeros_like(outage, dtype=np.int64)
    best_sector = np.zeros_like(outage, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        outage,
        x_axis,
        y_axis,
        test_locations,
        test_outages,
        bs_loc_km=C28_5_BS_LOC_KM,
        sector_boresite_deg=C28_5_SECTOR_BORESITE_DEG,
        bs_power_watt=np.asarray([0.1], dtype=np.float64),
        bs_power_watt_by_bs=np.full(C28_5_BS_LOC_KM.shape[0], 0.1, dtype=np.float64),
        active_original_bs_ids=np.arange(C28_5_BS_LOC_KM.shape[0], dtype=np.int64),
        height_map=height_map,
        best_bs=best_bs,
        best_sector=best_sector,
        uav_height_km=np.asarray([0.1], dtype=np.float64),
        max_height_m=np.asarray([90.0], dtype=np.float64),
        bs_downtilt_deg=np.asarray([100.0], dtype=np.float64),
        carrier_freq_ghz=np.asarray([2.0], dtype=np.float64),
        los_samples=np.asarray([100], dtype=np.int64),
        c28_5_map_contract=np.asarray(C28_5_MAP_CONTRACT),
        c28_5_legacy_map_sha256=np.asarray(C28_5_LEGACY_MAP_SHA256),
        best_link_labels_valid=np.asarray([0], dtype=np.int8),
    )
    prepared_core_hashes = {
        member: npz_member_sha256(output_path, member)
        for member in C28_5_MEMBER_SHA256
    }
    if prepared_core_hashes != C28_5_MEMBER_SHA256:
        raise RuntimeError("prepared C28.5 map changed the registered core arrays")

    manifest = {
        "contract": C28_5_MAP_CONTRACT,
        "legacy_source": str(legacy_path),
        "legacy_source_sha256": C28_5_LEGACY_MAP_SHA256,
        "legacy_member_sha256": member_hashes,
        "prepared_map": str(output_path),
        "prepared_archive_sha256": file_sha256(output_path),
        "prepared_semantic_sha256": semantic_npz_sha256(output_path),
        "prepared_core_member_sha256": prepared_core_hashes,
        "city_seed": C28_5_CITY_SEED,
        "grid_shape": [201, 201],
        "coordinate_scale_m": 10.0,
        "best_link_labels_valid": False,
    }
    resolved_manifest = manifest_path or output_path.with_suffix(".manifest.json")
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_c28_5_prepared_map(path: Path) -> None:
    path = Path(path)
    core_hashes = {
        member: npz_member_sha256(path, member)
        for member in C28_5_MEMBER_SHA256
    }
    if core_hashes != C28_5_MEMBER_SHA256:
        raise ValueError("radiomap does not contain the registered C28.5 core arrays")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != C28_5_PREPARED_FIELDS:
            raise ValueError("C28.5 prepared-map field set mismatch")
        contract = str(np.asarray(data.get("c28_5_map_contract", "")).reshape(()).item())
        source_hash = str(
            np.asarray(data.get("c28_5_legacy_map_sha256", "")).reshape(()).item()
        )
        if contract != C28_5_MAP_CONTRACT or source_hash != C28_5_LEGACY_MAP_SHA256:
            raise ValueError("radiomap is missing the registered C28.5 reconstruction metadata")
        if not np.array_equal(np.asarray(data["bs_loc_km"]), C28_5_BS_LOC_KM):
            raise ValueError("C28.5 base-station geometry mismatch")
        if not np.array_equal(
            np.asarray(data["sector_boresite_deg"]), C28_5_SECTOR_BORESITE_DEG
        ):
            raise ValueError("C28.5 sector geometry mismatch")
        if not np.array_equal(np.asarray(data["height_map"]), reconstruct_c28_5_height_map()):
            raise ValueError("C28.5 reconstructed city height map mismatch")
        outage = np.asarray(data["arr_0"])
        if not np.array_equal(np.asarray(data["best_bs"]), np.zeros_like(outage)):
            raise ValueError("C28.5 inactive best-BS compatibility labels mismatch")
        if not np.array_equal(np.asarray(data["best_sector"]), np.zeros_like(outage)):
            raise ValueError("C28.5 inactive best-sector compatibility labels mismatch")
        expected_scalars = {
            "bs_power_watt": 0.1,
            "uav_height_km": 0.1,
            "max_height_m": 90.0,
            "bs_downtilt_deg": 100.0,
            "carrier_freq_ghz": 2.0,
            "los_samples": 100,
            "best_link_labels_valid": 0,
        }
        for field, expected in expected_scalars.items():
            actual = np.asarray(data[field]).reshape(-1)[0].item()
            if actual != expected:
                raise ValueError(f"C28.5 prepared-map field mismatch: {field}")
        if not np.array_equal(
            np.asarray(data["bs_power_watt_by_bs"]),
            np.full(C28_5_BS_LOC_KM.shape[0], 0.1, dtype=np.float64),
        ):
            raise ValueError("C28.5 per-BS power metadata mismatch")
        if not np.array_equal(
            np.asarray(data["active_original_bs_ids"]),
            np.arange(C28_5_BS_LOC_KM.shape[0], dtype=np.int64),
        ):
            raise ValueError("C28.5 active BS identity metadata mismatch")
