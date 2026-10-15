from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wwm_nav.agent import WWMNavConfig, validate_c28_5_config
from wwm_nav.c28_5_map import (
    C28_5_LEGACY_MAP_SHA256,
    file_sha256,
    prepare_c28_5_radiomap,
    semantic_npz_sha256,
    validate_c28_5_prepared_map,
    verify_c28_5_legacy_map,
)
from wwm_nav.cli_config import load_config_file


REPOSITORY = Path(__file__).resolve().parents[1]
LEGACY_MAP = REPOSITORY / "data/c28_5/radioenvir_legacy.npz"
DEFAULT_CONFIG = REPOSITORY / "configs/wwm_nav_default.yaml"


def test_bundled_legacy_map_identity():
    assert file_sha256(LEGACY_MAP) == C28_5_LEGACY_MAP_SHA256
    verify_c28_5_legacy_map(LEGACY_MAP)


def test_prepared_map_is_semantically_stable_and_valid(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    prepare_c28_5_radiomap(LEGACY_MAP, first)
    prepare_c28_5_radiomap(LEGACY_MAP, second)
    validate_c28_5_prepared_map(first)
    validate_c28_5_prepared_map(second)
    assert semantic_npz_sha256(first) == semantic_npz_sha256(second)


def test_default_yaml_satisfies_strict_contract(tmp_path: Path):
    prepared = tmp_path / "c28_5.npz"
    prepare_c28_5_radiomap(LEGACY_MAP, prepared)
    values = load_config_file(DEFAULT_CONFIG)
    values["radiomap_path"] = prepared
    config = WWMNavConfig(**values)
    validate_c28_5_config(config)

    with pytest.raises(ValueError, match="strict C28.5 configuration mismatch"):
        validate_c28_5_config(replace(config, max_steps=299))
