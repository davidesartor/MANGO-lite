from __future__ import annotations
from typing import Protocol, NamedTuple, Any, Sequence
from itertools import chain
import torch
from . import spaces


# this is not a good way to type this version
# but it will minimize changes when addin support for generic types
ObsType = torch.Tensor
ActType = int


class Transition(NamedTuple):
    start_obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    terminated: bool
    truncated: bool
    steps: list[Transition] = []

    @classmethod
    def from_steps(cls, comand, steps: Sequence[Transition]) -> Transition:
        return Transition(
            start_obs=steps[0].start_obs,
            action=comand,
            next_obs=steps[-1].next_obs,
            reward=sum(step.reward for step in steps),
            terminated=steps[-1].terminated,
            truncated=steps[-1].truncated,
            steps=list(steps),
        )


class StackedTransitions(NamedTuple):
    start_obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class Trainer(Protocol):
    def train(self, transitions: StackedTransitions) -> float:
        ...


class Policy(Protocol):
    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...


class AbstractAction(Protocol):
    def mask(self, obs: ObsType) -> ObsType:
        ...

    def beta(self, transition: Transition) -> bool:
        ...

    def reward(self, transition: Transition) -> float:
        ...


class Environment(Protocol):
    @property
    def action_space(self) -> spaces.Discrete:
        ...

    @property
    def observation_space(self) -> spaces.Space:
        ...

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        ...
