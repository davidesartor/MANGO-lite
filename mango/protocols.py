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


class StackedTransitions(NamedTuple):
    start_obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class AbstractTransition(NamedTuple):
    start_obs: ObsType
    comand: ActType
    trajectory: list[ObsType]
    rewards: list[float]
    terminated: bool
    truncated: bool

    def tails(self) -> list[Transition]:
        rewards = [sum(self.rewards[i:]) for i in range(len(self.rewards))]
        term, trunc = self.terminated, self.truncated
        end_obs = self.trajectory[-1]
        return [
            Transition(start_obs, self.comand, end_obs, reward, term, trunc)
            for start_obs, reward in zip(self.trajectory[:-1], rewards)
        ]

    @classmethod
    def from_lower_steps(
        cls, comand: ActType, transitions: Sequence[AbstractTransition]
    ) -> AbstractTransition:
        if any(t.terminated or t.truncated for t in transitions[:-1]):
            raise ValueError("Only the last transition can be terminated or truncated")
        if any(t1.trajectory[-1] != t2.start_obs for t1, t2 in zip(transitions, transitions[1:])):
            raise ValueError("Each transition must start where the previous one ended")
        return cls(
            start_obs=transitions[0].start_obs,
            comand=comand,
            trajectory=list(chain.from_iterable(t.trajectory for t in transitions)),
            rewards=list(chain.from_iterable(t.rewards for t in transitions)),
            terminated=transitions[-1].terminated,
            truncated=transitions[-1].truncated,
        )


class Trainer(Protocol):
    def train(self, transitions: StackedTransitions) -> float:
        ...


class Policy(Protocol):
    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    @classmethod
    def make(cls, action_space: spaces.Discrete, **kwargs) -> tuple[Policy, Trainer]:
        ...


class AbstractAction(Protocol):
    def mask(self, obs: ObsType) -> ObsType:
        ...

    def beta(self, transition: AbstractTransition) -> bool:
        ...

    def reward(self, transition: AbstractTransition) -> float:
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
