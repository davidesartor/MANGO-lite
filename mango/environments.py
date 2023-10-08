from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, TypeVar
from gymnasium import spaces

ObsType = TypeVar("ObsType")


class Environment(Protocol[ObsType]):
    @property
    def observation_space(self) -> spaces.Space[ObsType]:
        ...

    @property
    def action_space(self) -> spaces.Discrete:
        ...

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, dict]:
        ...

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        ...


@dataclass(eq=False, slots=True)
class DummyEnvironment(Environment):
    state: int = 0
    observation_space: spaces.Space = spaces.Discrete(2)
    action_space: spaces.Discrete = spaces.Discrete(2)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        reward = float(self.state == action)
        self.state = self.observation_space.sample()
        return self.state, reward, False, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[int, dict]:
        self.state = self.observation_space.sample()
        return self.state, {}
