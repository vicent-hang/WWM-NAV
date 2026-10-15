from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wwm_nav.config import NavigationConfig
from wwm_nav.radiomap import RadioMap


ACTIONS = np.array(
    [
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0],
    ],
    dtype=np.int64,
)
ACTION_NAMES = ("north", "east", "south", "west")


@dataclass(frozen=True)
class StepResult:
    state: np.ndarray
    reward: float
    terminal: bool
    outbound: bool
    done: bool
    outage: float


class GridNavigationEnv:
    """UAV navigation environment backed by a precomputed radio map."""

    def __init__(
        self,
        radio_map: RadioMap,
        goal_m: np.ndarray | list[float] | tuple[float, float],
        config: NavigationConfig | None = None,
    ) -> None:
        self.radio_map = radio_map
        self.goal_m = np.asarray(goal_m, dtype=np.float64)
        self.config = config or NavigationConfig()
        self._step_count = 0
        self.state_m: np.ndarray | None = None

        if self.goal_m.shape != (2,):
            raise ValueError(f"expected goal shape (2,), got {self.goal_m.shape}")

    @property
    def observation_dim(self) -> int:
        return 5

    @property
    def action_size(self) -> int:
        return len(ACTIONS)

    def reset(
        self,
        start_m: np.ndarray | list[float] | tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        self._step_count = 0
        if start_m is None:
            start_m = self.sample_start(rng=rng)
        self.state_m = self.snap_to_map(np.asarray(start_m, dtype=np.float64))
        return self.state_m.copy()

    def sample_start(self, rng: np.random.Generator | None = None, margin_m: float = 50.0) -> np.ndarray:
        rng = rng or np.random.default_rng()
        x_min, x_max, y_min, y_max = self.radio_map.bounds_m
        start = np.array(
            [
                rng.uniform(x_min + margin_m, x_max - margin_m),
                rng.uniform(y_min + margin_m, y_max - margin_m),
            ],
            dtype=np.float64,
        )
        return self.snap_to_map(start)

    def snap_to_map(self, xy_m: np.ndarray) -> np.ndarray:
        return self.radio_map.index_to_xy(self.radio_map.nearest_index(xy_m))

    def observe(self, state_m: np.ndarray | None = None) -> np.ndarray:
        state = self.state_m if state_m is None else np.asarray(state_m, dtype=np.float64)
        if state is None:
            raise RuntimeError("environment has not been reset")

        x_min, x_max, y_min, y_max = self.radio_map.bounds_m
        scale = np.array([x_max - x_min, y_max - y_min], dtype=np.float64)
        xy_norm = (state - np.array([x_min, y_min], dtype=np.float64)) / scale
        goal_delta_norm = (self.goal_m - state) / scale
        outage = np.array([self._outage_at_xy(state)], dtype=np.float64)
        return np.concatenate((xy_norm, goal_delta_norm, outage)).astype(np.float32)

    def step_action(self, action_idx: int) -> StepResult:
        if self.state_m is None:
            raise RuntimeError("environment has not been reset")
        result = self.step(self.state_m, action_idx)
        self.state_m = result.state.copy()
        return result

    def step(self, state_m: np.ndarray, action_idx: int) -> StepResult:
        if not 0 <= action_idx < len(ACTIONS):
            raise ValueError(f"invalid action index {action_idx}")

        self._step_count += 1
        current_state = np.asarray(state_m, dtype=np.float64)
        dx_m, dy_m = self.radio_map.resolution_m
        displacement = np.array([ACTIONS[action_idx, 0] * dx_m, ACTIONS[action_idx, 1] * dy_m])
        next_state = current_state + displacement

        x_min, x_max, y_min, y_max = self.radio_map.bounds_m
        outbound = bool(
            next_state[0] < x_min
            or next_state[0] > x_max
            or next_state[1] < y_min
            or next_state[1] > y_max
        )
        next_state[0] = np.clip(next_state[0], x_min, x_max)
        next_state[1] = np.clip(next_state[1], y_min, y_max)

        previous_distance = np.linalg.norm(current_state - self.goal_m)
        next_distance = np.linalg.norm(next_state - self.goal_m)
        terminal = bool(next_distance <= self.config.goal_tolerance_m)
        outage = self._outage_at_xy(next_state)

        reward = -self.config.move_penalty
        reward -= self.config.outage_penalty * outage
        if outage >= self.config.high_outage_threshold:
            reward -= self.config.high_outage_reward_penalty
        step_scale = max(float(np.mean(self.radio_map.resolution_m)), 1.0)
        reward += self.config.distance_shaping_weight * (previous_distance - next_distance) / step_scale
        if terminal:
            reward += self.config.terminal_reward
        if outbound:
            reward -= self.config.out_of_bounds_penalty

        done = terminal or outbound or self._step_count >= self.config.max_steps
        return StepResult(
            state=next_state,
            reward=float(reward),
            terminal=terminal,
            outbound=outbound,
            done=done,
            outage=float(outage),
        )

    def _outage_at_xy(self, state_m: np.ndarray) -> float:
        return self.radio_map.outage_at_xy(state_m)

    def action_between(self, current_m: np.ndarray, next_m: np.ndarray) -> int:
        delta = np.asarray(next_m, dtype=np.float64) - np.asarray(current_m, dtype=np.float64)
        dx_m, dy_m = self.radio_map.resolution_m
        normalized = np.array([round(delta[0] / dx_m), round(delta[1] / dy_m)], dtype=np.int64)
        matches = np.where((ACTIONS == normalized).all(axis=1))[0]
        if matches.size != 1:
            raise ValueError(f"states are not one grid action apart: {current_m} -> {next_m}")
        return int(matches[0])

    def rollout_path(self, path_m: np.ndarray) -> dict[str, np.ndarray]:
        if path_m.ndim != 2 or path_m.shape[1] != 2:
            raise ValueError(f"expected path shape (N, 2), got {path_m.shape}")
        if len(path_m) < 2:
            raise ValueError("path must contain at least two states")

        self.reset(start_m=path_m[0])
        states = [path_m[0]]
        actions: list[int] = []
        rewards: list[float] = []
        outages: list[float] = []
        terminals: list[bool] = []

        current = path_m[0]
        for next_state in path_m[1:]:
            action_idx = self.action_between(current, next_state)
            result = self.step_action(action_idx)
            actions.append(action_idx)
            rewards.append(result.reward)
            outages.append(result.outage)
            terminals.append(result.terminal)
            states.append(result.state)
            current = result.state
            if result.done:
                break

        return {
            "states": np.asarray(states, dtype=np.float64),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "outages": np.asarray(outages, dtype=np.float64),
            "terminals": np.asarray(terminals, dtype=np.bool_),
        }
