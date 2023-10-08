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
