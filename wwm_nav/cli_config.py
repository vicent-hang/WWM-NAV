from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from wwm_nav.agent import WWMNavConfig


def _coerce_value(name: str, value: Any) -> Any:
    if name in {"radiomap_path", "save_dir", "resume_checkpoint_path"}:
        if value is None:
            return None
        return Path(value)
    if name in {"start_m", "goal_m", "actor_hidden_dims", "q_hidden_dims"}:
        return tuple(value)
    return value


def load_config_file(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a mapping, got {type(raw).__name__}")
    aliases = {
        "radiomap": "radiomap_path",
        "start": "start_m",
        "goal": "goal_m",
        "resume": "resume_checkpoint_path",
    }
    valid = {field.name for field in fields(WWMNavConfig)}
    result: dict[str, Any] = {}
    for raw_key, raw_value in raw.items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key == "method":
            continue
        if key not in valid:
            raise ValueError(f"unknown WWMNavConfig field in {path}: {raw_key}")
        result[key] = _coerce_value(key, raw_value)
    return result


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result
