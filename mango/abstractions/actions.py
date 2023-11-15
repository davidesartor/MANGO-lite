from typing import Any, Mapping, Protocol, Sequence, TypeVar
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import gymnasium as gym

ObsType = TypeVar("ObsType", bound=npt.NDArray)


class AbstractActions(Protocol[ObsType]):
    action_space: gym.spaces.Discrete
    
    def beta(self, start_state: ObsType, next_state: ObsType) -> float:
        ...

    def compatibility(
        self, action: int, start_state: ObsType, next_state: ObsType
    ) -> float:
        ...
