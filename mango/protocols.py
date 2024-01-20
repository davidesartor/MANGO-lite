from __future__ import annotations
from typing import Protocol, NamedTuple, Any, Sequence, SupportsFloat
from itertools import chain
import torch
from . import spaces


# this is not a good way to type this version
# but it will minimize changes when addin support for generic types
ObsType = torch.Tensor
ActType = int


class Transition(NamedTuple):
    start_obs: ObsType
    action: ActType | None
    next_obs: ObsType
    reward: float
    terminated: bool
    truncated: bool
    steps: list[Transition] = []

    @property
    def trajectory(self) -> list[ObsType]:
        if not self.steps:
            return [self.start_obs, self.next_obs]
        return [self.start_obs] + list(
            chain.from_iterable(step.trajectory[1:] for step in self.steps)
        )

    @classmethod
    def from_steps(cls, comand: ActType | None, steps: Sequence[Transition]) -> Transition:
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

    def to(self, device: str, non_blocking=False) -> StackedTransitions:
        return StackedTransitions(*(el.to(device, non_blocking=non_blocking) for el in self))


class TrainInfo(NamedTuple):
    loss: torch.Tensor
    td: torch.Tensor


class Trainer(Protocol):
    def train(self, transitions: StackedTransitions) -> TrainInfo:
        ...


class Policy(Protocol):
    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...


class AbstractAction(Protocol):
    def mask(self, obs: ObsType) -> ObsType:
        ...

    def beta(self, transition: Sequence[Transition]) -> bool:
        ...

    def reward(self, transition: Sequence[Transition]) -> float:
        ...


class Environment(Protocol):
    @property
    def action_space(self) -> spaces.Discrete:
        ...

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        ...
