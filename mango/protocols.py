from __future__ import annotations
from typing import Protocol, NamedTuple, Any, Sequence
import numpy as np
import numpy.typing as npt
import torch
from . import spaces


# this is not a good way to type this version
# but it will minimize changes when addin support for generic types
ObsType = npt.NDArray[np.floating[Any] | np.integer[Any]]
ActType = np.integer[Any] | int
OptionType = ActType | tuple[int, ActType]


class Transition(NamedTuple):
    start_obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = {}


class OptionTransition(NamedTuple):
    comand: ActType
    trajectory: list[ObsType]
    rewards: list[float]
    failed: bool
    beta: bool
    terminated: bool
    truncated: bool

    def all_transitions(self) -> list[Transition]:
        rewards = [sum(self.rewards[i:]) for i in range(len(self.rewards))]
        return [
            Transition(
                start_obs=start_obs,
                action=self.comand,
                next_obs=self.trajectory[-1],
                reward=reward,
                terminated=self.terminated,
                truncated=self.truncated,
            )
            for start_obs, reward in zip(self.trajectory[:-1], rewards)
        ]

    def flatten(self) -> Transition:
        return Transition(
            start_obs=self.trajectory[0],
            action=self.comand,
            next_obs=self.trajectory[-1],
            reward=sum(self.rewards),
            terminated=self.terminated,
            truncated=self.truncated,
        )


class TrainInfo(NamedTuple):
    loss: float
    td: npt.NDArray[np.floating[Any]]


class AbstractActions(Protocol):
    action_space: spaces.Discrete

    def mask(self, comand: ActType, obs: ObsType) -> ObsType:
        ...

    def beta(self, comand: ActType, transition: Transition) -> bool:
        ...

    def reward(self, comand: ActType, transition: Transition) -> float:
        ...

    def has_failed(self, comand: ActType, start_obs: ObsType, next_obs: ObsType) -> bool:
        ...


class Policy(Protocol):
    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    def train(self, transitions: list[Transition]) -> TrainInfo:
        ...

    @classmethod
    def make(cls, action_space: spaces.Discrete, **kwargs) -> Policy:
        ...


class DynamicPolicy(Protocol):
    def get_action(self, comand: ActType, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    def train(self, comand: ActType, transitions: list[Transition]) -> TrainInfo:
        ...

    @classmethod
    def make(
        cls, comand_space: spaces.Discrete, action_space: spaces.Discrete, **kwargs
    ) -> DynamicPolicy:
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
