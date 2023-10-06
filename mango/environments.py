from __future__ import annotations
from dataclasses import dataclass, astuple
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, TypeVar
from typing import Generic, NamedTuple
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

ObsType = TypeVar("ObsType")
AbsObsType = TypeVar("AbsObsType")
ActType = TypeVar("ActType")
AbsActType = TypeVar("AbsActType")


class Environment(Protocol[ObsType, ActType]):
    @property
    def observation_space(self) -> spaces.Space[ObsType]:
        ...

    @property
    def action_space(self) -> spaces.Space[ActType]:
        ...

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        ...

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        ...


@dataclass(eq=False, slots=True)
class DummyEnvironment(Environment):
    state: int = 0

    @property
    def observation_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        return 1 - action, float(self.state == action), False, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[int, dict]:
        self.state = 0
        return self.state, {}
