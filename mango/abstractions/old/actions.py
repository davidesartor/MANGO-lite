from typing import Any, Protocol, TypeVar
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import numpy.typing as npt

ObsType = TypeVar("ObsType")


class ActionCompatibility(Protocol):
    action_space: gym.spaces.Discrete

    def __call__(
        self, comand: int, start_state: npt.NDArray, next_state: npt.NDArray
    ) -> float:
        ...


@dataclass(frozen=True, eq=False)
class FullCompatibility(ActionCompatibility):
    action_space: gym.spaces.Discrete

    def __call__(self, comand: Any, start_state: Any, next_state: Any) -> float:
        return 1.0


@dataclass(frozen=True, eq=False)
class GridCompatibility(ActionCompatibility):
    action_space = gym.spaces.Discrete(4)

    def __call__(
        self, comand: int, start_state: npt.NDArray, next_state: npt.NDArray
    ) -> float:
        LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
        delta_y, delta_x = next_state - start_state

        if delta_y == 0 and delta_x == -1:
            executed = LEFT
        elif delta_y == 1 and delta_x == 0:
            executed = DOWN
        elif delta_y == 0 and delta_x == 1:
            executed = RIGHT
        elif delta_y == -1 and delta_x == 0:
            executed = UP
        else:
            executed = None

        if executed == comand:
            return 1.0
        elif delta_y == 0 and delta_x == 0:
            return -0.1
        else:
            return -1.0


@dataclass(frozen=True, eq=False)
class CondensationCompatibility(GridCompatibility):
    action_space = gym.spaces.Discrete(4)
    channel: int = 0

    def __call__(
        self, comand: int, start_state: npt.NDArray, next_state: npt.NDArray
    ) -> float:
        grid_shape = start_state.shape[:-1]
        start_idx = np.argmax(start_state[:, :, self.channel])
        next_idx = np.argmax(next_state[:, :, self.channel])
        y_start, x_start = np.unravel_index(start_idx, grid_shape)
        y_next, x_next = np.unravel_index(next_idx, grid_shape)
        return super().__call__(
            comand, np.array(y_start, x_start), np.array(y_next, x_next)
        )
