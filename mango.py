from __future__ import annotations
from typing import Any, Generic, TypeVar
import numpy as np

from . dynamicpolicies import DQnetPolicyMapper
from . protocols import Concept, Environment


ObsType = TypeVar("ObsType")


class Mango(Generic[ObsType]):
    def __init__(
        self,
        environment: Environment[ObsType, int],
        concepts: list[Concept[ObsType, np.ndarray]],
        
    ) -> None:
        ...
        

    def reset(self) -> None:
        self.layers[-1].reset()
