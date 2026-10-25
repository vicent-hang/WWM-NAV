from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationConfig:
    """Shared settings for grid-based UAV navigation."""

    x_max_m: float = 2000.0
    y_max_m: float = 2000.0
    goal_tolerance_m: float = 30.0
    move_penalty: float = 1.0
    outage_penalty: float = 10.0
    high_outage_threshold: float = 0.45
    high_outage_reward_penalty: float = 0.0
    terminal_reward: float = 200.0
    out_of_bounds_penalty: float = 10000.0
    distance_shaping_weight: float = 2.0
    max_steps: int = 300
