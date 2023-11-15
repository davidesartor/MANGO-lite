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
        delta_y, delta_x = next_state - start_state

        LEFT, DOWN, RIGHT, UP, STAY = 0, 1, 2, 3, -1
        transition = {
            (0, -1): LEFT,
            (1, 0): DOWN,
            (0, 1): RIGHT,
            (-1, 0): UP,
            (0, 0): STAY,
        }.get((delta_y, delta_x), None)

        if transition == comand:
            return 1.0
        elif transition == STAY:
            return -0.1
        else:
            return -1.0


class CondensationCompatibility(GridCompatibility):
    def __call__(
        self, comand: int, start_state: npt.NDArray, next_state: npt.NDArray
    ) -> float:
        grid_shape = start_state.shape[:-1]
        start_idx = np.argmax(start_state[:, :, -1])
        next_idx = np.argmax(next_state[:, :, -1])
        y_start, x_start = np.unravel_index(start_idx, grid_shape)
        y_next, x_next = np.unravel_index(next_idx, grid_shape)
        return super().__call__(
            comand, np.array(y_start, x_start), np.array(y_next, x_next)
        )
