from __future__ import annotations
from typing import Protocol, NamedTuple, Any
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
    info: dict[str, Any]


class TensorTransitionLists(NamedTuple):
    start_obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: list[dict[str, Any]]


class AbstractActions(Protocol):
    action_space: spaces.Discrete

    def mask(self, comand: ActType, obs: ObsType) -> ObsType:
        ...

    def beta(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> tuple[bool, bool]:
        ...

    def compatibility(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> float:
        ...


class Policy(Protocol):
    def get_action(self, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    def train(self, transitions: TensorTransitionLists) -> float | None:
        ...

    @classmethod
    def make(cls, action_space: spaces.Discrete, **kwargs) -> Policy:
        ...


class DynamicPolicy(Protocol):
    def get_action(self, comand: ActType, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    def train(self, comand: ActType, transitions: TensorTransitionLists) -> float | None:
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

    @property
    def obs(self) -> ObsType:
        ...

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        ...
