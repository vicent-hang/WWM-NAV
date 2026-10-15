from __future__ import annotations

import argparse
from pathlib import Path

from wwm_nav.c28_5_map import prepare_c28_5_radiomap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the physics-complete radio map for the registered C28.5 experiment."
    )
    parser.add_argument(
        "--legacy-map",
        type=Path,
        default=Path("data/c28_5/radioenvir_legacy.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/c28_5/radioenvir_c28_5.npz"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_c28_5_radiomap(args.legacy_map, args.output, args.manifest)
    print(f"prepared C28.5 radio map: {manifest['prepared_map']}")
    print(f"legacy source SHA-256: {manifest['legacy_source_sha256']}")
    print(f"prepared archive SHA-256: {manifest['prepared_archive_sha256']}")
    print(f"prepared semantic SHA-256: {manifest['prepared_semantic_sha256']}")


if __name__ == "__main__":
    main()
