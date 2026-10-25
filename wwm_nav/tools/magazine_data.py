from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from wwm_nav.radiomap import RadioMap

WWM_LA_NAV = "WWM-LA-NAV"
METHODS = ("Direct RL", "SNARM", WWM_LA_NAV)
DISPLAY_NAMES = {"Direct RL": "Direct RL", "SNARM": "SNARM", WWM_LA_NAV: "WWM-NAV"}
COLORS = {"Direct RL": "#0072B2", "SNARM": "#009E73", WWM_LA_NAV: "#D55E00"}
LINE_STYLES = {"Direct RL": "-", "SNARM": "--", WWM_LA_NAV: "-."}
MARKERS = {"Direct RL": "o", "SNARM": "s", WWM_LA_NAV: "^"}
MARKER_OFFSETS = {"Direct RL": 0, "SNARM": 3, WWM_LA_NAV: 6}
EXPECTED_CONTRACT = "paired_common_map_navigation_journal_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_provenance(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("analysis_contract") != EXPECTED_CONTRACT:
        raise ValueError(
            f"expected analysis contract {EXPECTED_CONTRACT!r}, "
            f"got {provenance.get('analysis_contract')!r}"
        )
    provenance["_data_dir"] = str(data_dir.resolve())
    return provenance


def validate_inputs(provenance: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    data_dir = Path(provenance["_data_dir"])
    radio_map_path = data_dir / provenance["radiomap"]
    if sha256_file(radio_map_path) != provenance["radiomap_sha256"]:
        raise ValueError("radio-map SHA-256 does not match reference provenance")
    directories = {
        method: data_dir / provenance["methods"][method]
        for method in METHODS
    }
    for method, directory in directories.items():
        metrics_path = directory / "trial_metrics.csv"
        pairs_path = directory / "trial_pairs.csv"
        if sha256_file(metrics_path) != provenance["trial_metrics_sha256"][method]:
            raise ValueError(f"{method}: trial_metrics.csv SHA-256 mismatch")
        if sha256_file(pairs_path) != provenance["trial_pairs_sha256"][method]:
            raise ValueError(f"{method}: trial_pairs.csv SHA-256 mismatch")
    return radio_map_path, directories


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_method(
    method: str,
    directory: Path,
    radio_map: RadioMap,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]], float]:
    source_rows = _read_csv(directory / "trial_metrics.csv")
    rows: list[dict[str, Any]] = []
    rollouts: list[dict[str, np.ndarray]] = []
    absolute_differences: list[np.ndarray] = []
    for expected_trial, source in enumerate(source_rows):
        trial = int(source["trial"])
        if trial != expected_trial:
            raise ValueError(f"{method}: expected ordered trial {expected_trial}, got {trial}")
        rollout_path = directory / "rollouts" / f"trial_{trial:04d}.npz"
        with np.load(rollout_path, allow_pickle=False) as data:
            rollout = {key: np.asarray(data[key]) for key in data.files}
        states = np.asarray(rollout["states"], dtype=np.float64)
        steps = int(float(source["steps"]))
        if states.ndim != 2 or states.shape[1] < 2 or states.shape[0] != steps + 1:
            raise ValueError(f"{rollout_path}: states do not match the recorded step count")
        rescored = np.asarray(
            [radio_map.outage_at_xy(position[:2]) for position in states[1:]],
            dtype=np.float64,
        )
        saved_outages = np.asarray(rollout.get("outages", []), dtype=np.float64).reshape(-1)
        if saved_outages.size == rescored.size:
            absolute_differences.append(np.abs(saved_outages - rescored))
        start = np.asarray([float(source["start_x"]), float(source["start_y"])])
        goal = np.asarray([float(source["goal_x"]), float(source["goal_y"])])
        if not np.array_equal(states[0, :2], start):
            raise ValueError(f"{method} trial {trial}: rollout start does not match CSV")
        initial_distance = float(np.linalg.norm(start - goal))
        path_length = float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum())
        rows.append(
            {
                "trial": trial,
                "method": method,
                "start_x": float(start[0]),
                "start_y": float(start[1]),
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "initial_distance_m": initial_distance,
                "success": int(float(source["success"])),
                "steps": steps,
                "path_length_m": path_length,
            }
        )
        rollouts.append(
            {"states": states[:, :2], "outages": rescored, "goal": goal, "start": start}
        )
    differences = np.concatenate(absolute_differences) if absolute_differences else np.empty(0)
    mae = float(np.mean(differences)) if differences.size else float("nan")
    return rows, rollouts, mae


def validate_pairing(tables: dict[str, list[dict[str, Any]]]) -> None:
    reference = tables[METHODS[0]]
    if len(reference) != 100:
        raise ValueError(f"expected 100 trials, got {len(reference)}")
    keys = ("trial", "start_x", "start_y", "goal_x", "goal_y", "initial_distance_m")
    for method in METHODS[1:]:
        rows = tables[method]
        if len(rows) != len(reference):
            raise ValueError(f"{method}: trial count mismatch")
        for left, right in zip(reference, rows, strict=True):
            for key in keys:
                if not np.isclose(float(left[key]), float(right[key]), atol=1e-9, rtol=0.0):
                    raise ValueError(f"{method} trial {left['trial']}: paired field {key} differs")


def goal_reaching_curve(
    rows: list[dict[str, Any]], indices: np.ndarray, max_steps: int
) -> np.ndarray:
    steps = np.asarray([rows[index]["steps"] for index in indices], dtype=np.float64)
    success = np.asarray([rows[index]["success"] for index in indices], dtype=bool)
    budgets = np.arange(max_steps + 1, dtype=np.float64)
    return np.mean(success[:, None] & (steps[:, None] <= budgets[None, :]), axis=0)


def cumulative_outage_curves(
    rollouts: list[dict[str, np.ndarray]], max_steps: int
) -> np.ndarray:
    curves = np.zeros((len(rollouts), max_steps + 1), dtype=np.float64)
    for trial, rollout in enumerate(rollouts):
        cumulative = np.cumsum(np.asarray(rollout["outages"], dtype=np.float64))
        curves[trial, 1 : cumulative.size + 1] = cumulative
        curves[trial, cumulative.size + 1 :] = cumulative[-1]
    return curves
