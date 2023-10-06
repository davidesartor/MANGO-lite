from __future__ import annotations
from typing import Any, Generic, NewType, TypeVar
import numpy as np
import numpy.typing as npt

from actions import ActionCompatibility

from .dynamicpolicies import DQnetPolicyMapper
from .utils import Environment


ObsType = TypeVar("ObsType")


class Mango(Generic[ObsType]):
    def __init__(
        self,
        environment: Environment[ObsType, int],
        concepts: list[Concept[ObsType, npt.NDArray, int]],
        actions: list[ActionCompatibility[int, npt.NDArray]],
    ) -> None:
        ...

    def reset(self) -> None:
        ...
