from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wwm_nav.agent import (
    EXPERIMENT_CONTRACT,
    WWMNavConfig,
    serializable_config,
    validate_wwm_nav_config,
)
from wwm_nav.wwm_nav_map import file_sha256, semantic_npz_sha256
from wwm_nav.cli_config import load_config_file
from wwm_nav.radiomap import RadioMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the map and configuration for the WWM-NAV run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wwm_nav_default.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WWMNavConfig(**load_config_file(args.config))
    validate_wwm_nav_config(config)
    radio_map = RadioMap.from_npz(config.radiomap_path, require_physics=True)
    config_payload = serializable_config(config)
    config_digest = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    print(f"experiment contract: {EXPERIMENT_CONTRACT}")
    print(f"config: {args.config}")
    print(f"config SHA-256: {config_digest}")
    print(f"map: {config.radiomap_path}")
    print(f"map archive SHA-256: {file_sha256(config.radiomap_path)}")
    print(f"map semantic SHA-256: {semantic_npz_sha256(config.radiomap_path)}")
    print(f"map shape: {radio_map.outage.shape}")
    print(f"map bounds (m): {radio_map.bounds_m}")
    print(f"map resolution (m): {radio_map.resolution_m}")
    print("WWM-NAV setup validation: OK")


if __name__ == "__main__":
    main()
