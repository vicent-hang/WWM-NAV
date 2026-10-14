from __future__ import annotations

from pathlib import Path

from wwm_nav.cli_config import load_config_file


def test_yaml_config_aliases(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("radiomap: data/map.npz\nstart_m: [1.0, 2.0]\ngoal: [3.0, 4.0]\nepisodes: 2\n")
    data = load_config_file(path)
    assert data["radiomap_path"] == Path("data/map.npz")
    assert data["start_m"] == (1.0, 2.0)
    assert data["goal_m"] == (3.0, 4.0)
    assert data["episodes"] == 2
