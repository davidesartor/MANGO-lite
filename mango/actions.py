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
        print(start_state, next_state)
        if np.all(next_state == start_state):
            return -0.1

        delta_y, delta_x = next_state - start_state

        # 0 = left
        if comand == 0:
            if delta_y == 0 and delta_x == -1:
                return 1.0
        # 1 = down
        elif comand == 1:
            if delta_y == 1 and delta_x == 0:
                return 1.0
        # 2 = right
        elif comand == 2:
            if delta_y == 0 and delta_x == 1:
                return 1.0
        # 3 = up
        elif comand == 3:
            if delta_y == -1 and delta_x == 0:
                return 1.0
        return -1.0
    
@dataclass(frozen=True, eq=False)
class CondensationCompatibility(ActionCompatibility):
    action_space = gym.spaces.Discrete(4)

    def __call__(
        self, comand: int, start_state: npt.NDArray, next_state: npt.NDArray
    ) -> float:

        if np.all(next_state == start_state):
            return -0.1

        x_start, y_start = np.unravel_index(np.argmax(start_state[:,:,-1]), start_state[:,:,-1].shape)
        x_end, y_end = np.unravel_index(np.argmax(next_state[:,:,-1]), next_state[:,:,-1].shape)
        delta_y, delta_x = x_end - x_start, y_end - y_start
        # 0 = left
        if comand == 0:
            if delta_y == 0 and delta_x == -1:
                return 1.0
        # 1 = down
        elif comand == 1:
            if delta_y == 1 and delta_x == 0:
                return 1.0
        # 2 = right
        elif comand == 2:
            if delta_y == 0 and delta_x == 1:
                return 1.0
        # 3 = up
        elif comand == 3:
            if delta_y == -1 and delta_x == 0:
                return 1.0
        return -1.0
