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
        self._before_step_outage_query()
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

    def _before_step_outage_query(self) -> None:
        """Hook for stateful radio processes; static environments do nothing."""

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


class DynamicInterferenceNavigationEnv(GridNavigationEnv):
    """Hidden seven-state interference process for the opt-in dynamic suite.

    The process advances immediately before the outage measurement on steps
    4, 8, 12, ... . Observation calls are pure and never advance the process.
    """

    def __init__(
        self,
        radio_map: RadioMap,
        goal_m: np.ndarray | list[float] | tuple[float, float],
        config: NavigationConfig | None = None,
        *,
        interference_seed: int | np.random.SeedSequence,
    ) -> None:
        super().__init__(radio_map, goal_m, config)
        if not radio_map.has_dynamic_interference:
            raise ValueError("dynamic interference environment requires a dynamic RadioMap")
        assert radio_map.transition_stride is not None
        assert radio_map.interference_update_period_steps is not None
        assert radio_map.interference_move_probability is not None
        self.transition_stride = int(radio_map.transition_stride)
        self.update_period_steps = int(radio_map.interference_update_period_steps)
        self.move_probability = float(radio_map.interference_move_probability)
        self._interference_seed = interference_seed
        self._interference_rng = np.random.default_rng(interference_seed)
        self.interference_state_id: int | None = None
        self.interference_state_history: list[int] = []
        self.interference_transition_history: list[bool] = []

    def reset(
        self,
        start_m: np.ndarray | list[float] | tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        self._interference_rng = np.random.default_rng(self._interference_seed)
        assert self.radio_map.dynamic_outage_states is not None
        self.interference_state_id = int(
            self._interference_rng.integers(self.radio_map.dynamic_outage_states.shape[0])
        )
        self.interference_state_history = [self.interference_state_id]
        self.interference_transition_history = []
        return super().reset(start_m=start_m, rng=rng)

    def _before_step_outage_query(self) -> None:
        if self.interference_state_id is None:
            raise RuntimeError("environment has not been reset")
        transitioned = False
        if self._step_count % self.update_period_steps == 0:
            transitioned = bool(self._interference_rng.random() < self.move_probability)
            if transitioned:
                assert self.radio_map.dynamic_outage_states is not None
                self.interference_state_id = (
                    self.interference_state_id + self.transition_stride
                ) % self.radio_map.dynamic_outage_states.shape[0]
        self.interference_transition_history.append(transitioned)
        self.interference_state_history.append(self.interference_state_id)

    def _outage_at_xy(self, state_m: np.ndarray) -> float:
        if self.interference_state_id is None:
            raise RuntimeError("environment has not been reset")
        return self.radio_map.dynamic_outage_at_xy(self.interference_state_id, state_m)

    def dynamic_state_dict(self) -> dict[str, object]:
        if self.interference_state_id is None:
            raise RuntimeError("environment has not been reset")
        seed_entropy = (
            self._interference_seed.entropy
            if isinstance(self._interference_seed, np.random.SeedSequence)
            else int(self._interference_seed)
        )
        return {
            "interference_seed_entropy": seed_entropy,
            "interference_state_id": self.interference_state_id,
            "step_count": int(self._step_count),
            "rng_state": self._interference_rng.bit_generator.state,
            "state_m": None if self.state_m is None else self.state_m.tolist(),
            "interference_state_history": list(self.interference_state_history),
            "interference_transition_history": list(
                self.interference_transition_history
            ),
        }

    def load_dynamic_state_dict(self, state: dict[str, object]) -> None:
        state_id = int(state.get("interference_state_id", -1))
        step_count = int(state.get("step_count", -1))
        assert self.radio_map.dynamic_outage_states is not None
        state_count = int(self.radio_map.dynamic_outage_states.shape[0])
        if not 0 <= state_id < state_count or step_count < 0:
            raise ValueError("invalid dynamic interference resume state")
        expected_entropy = (
            self._interference_seed.entropy
            if isinstance(self._interference_seed, np.random.SeedSequence)
            else int(self._interference_seed)
        )
        stored_entropy = state.get("interference_seed_entropy")
        if isinstance(expected_entropy, (list, tuple, np.ndarray)):
            entropy_matches = np.array_equal(
                np.asarray(stored_entropy), np.asarray(expected_entropy)
            )
        else:
            entropy_matches = stored_entropy == expected_entropy
        if not entropy_matches:
            raise ValueError("dynamic interference resume seed mismatch")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError("dynamic interference resume RNG state is missing")
        state_m = np.asarray(state.get("state_m"), dtype=np.float64)
        if state_m.shape != (2,) or not np.all(np.isfinite(state_m)):
            raise ValueError("dynamic interference resume navigation state is invalid")
        if not np.array_equal(self.snap_to_map(state_m), state_m):
            raise ValueError("dynamic interference resume navigation state is off-grid")
        state_history = [
            int(value) for value in state.get("interference_state_history", [])
        ]
        transition_history = [
            bool(value) for value in state.get("interference_transition_history", [])
        ]
        if (
            len(state_history) != step_count + 1
            or len(transition_history) != step_count
            or not state_history
            or state_history[-1] != state_id
            or any(not 0 <= value < state_count for value in state_history)
        ):
            raise ValueError("dynamic interference resume history is inconsistent")
        validation_rng = np.random.default_rng(self._interference_seed)
        expected_state_id = int(validation_rng.integers(state_count))
        expected_state_history = [expected_state_id]
        expected_transition_history: list[bool] = []
        for step in range(1, step_count + 1):
            transitioned = False
            if step % self.update_period_steps == 0:
                transitioned = bool(validation_rng.random() < self.move_probability)
                if transitioned:
                    expected_state_id = (
                        expected_state_id + self.transition_stride
                    ) % state_count
            expected_transition_history.append(transitioned)
            expected_state_history.append(expected_state_id)
        if (
            state_history != expected_state_history
            or transition_history != expected_transition_history
            or rng_state != validation_rng.bit_generator.state
        ):
            raise ValueError("dynamic interference resume process state is inconsistent")
        self.interference_state_id = state_id
        self._step_count = step_count
        self.state_m = state_m.copy()
        self._interference_rng.bit_generator.state = rng_state
        self.interference_state_history = state_history
        self.interference_transition_history = transition_history
