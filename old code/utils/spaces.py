from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from functools import cache, cached_property
from itertools import chain
import math
import random
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Protocol,
    TypeVar,
    runtime_checkable,
)
import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass(frozen=True)
class SASRI(Generic[ObsType, ActType]):
    start_state: ObsType
    action: ActType
    end_state: ObsType
    reward: float
    truncated: bool
    terminated: bool
    info: dict[str, Any]

    preprocessed: SASRI[Any, Any] | None = None

    def preprocess(self, processing_function: Callable[[SASRI[ObsType, ActType]], SASRI[Any, Any]]) -> None:
        object.__setattr__(self, "preprocessed", processing_function(self))

    @staticmethod
    def from_sasri_template(
        template: SASRI[Any, Any],
        start_state: ObsType | None = None,
        action: ActType | None = None,
        end_state: ObsType | None = None,
        reward: float | None = None,
        truncated: bool | None = None,
        terminated: bool | None = None,
        info: dict[str, Any] | None = None,
        preprocessed: SASRI[Any, Any] | None = None,
    ) -> SASRI[Any, Any]:
        return SASRI(
            start_state if start_state is not None else template.start_state,
            action if action is not None else template.action,
            end_state if end_state is not None else template.end_state,
            reward if reward is not None else template.reward,
            truncated if truncated is not None else template.truncated,
            terminated if terminated is not None else template.terminated,
            info if info is not None else template.info,
            preprocessed if preprocessed is not None else template.preprocessed,
        )


T = TypeVar("T")


class SpaceType(Protocol[T]):
    def __contains__(self, item: T) -> bool:
        return self.to_gym().__contains__(item)

    def sample(self, mask: Any = None) -> T:
        return self.to_gym().sample(mask)

    def to_gym(self) -> gym.spaces.Space:
        ...


@runtime_checkable
class FlattenableSpace(SpaceType[T], Protocol[T]):
    @cached_property
    def flat_dim(self) -> int:
        return gym.spaces.flatdim(self.to_gym())

    @cache
    def to_flat_box(self) -> gym.spaces.Box:
        flat_box = gym.spaces.flatten_space(self.to_gym())
        if not isinstance(flat_box, gym.spaces.Box):
            raise TypeError(f"Space is not flattenable to Box: {self}")
        return flat_box

    def flatten_obs(self, observation: T) -> np.ndarray:
        flat_obs = gym.spaces.flatten(self.to_gym(), observation)
        if not isinstance(flat_obs, np.ndarray):
            raise TypeError(f"Observation is not flattenable to ndarray: {observation}")
        return flat_obs


T = TypeVar("T")


@dataclass(frozen=True, repr=False)
class FiniteSpace(list, FlattenableSpace[T]):
    elements: InitVar[Iterable[T] | int]

    def __post_init__(self, elements) -> None:
        if isinstance(elements, int):
            elements = range(elements)
        super().__init__(elements)

    def __repr__(self) -> str:
        return self.__class__.__qualname__ + f"({super().__repr__()})"

    def to_gym(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self))

    def sample(self, mask: Any = None) -> T:
        return random.choice(self)

    def flatten_obs(self, observation: T) -> np.ndarray:
        flat_obs = np.zeros(len(self))
        flat_obs[self.index(observation)] = 1
        return flat_obs


@dataclass(frozen=True)
class TensorSpace(FlattenableSpace[np.ndarray]):
    shape: tuple[int, ...]
    low: float | np.ndarray = field(default=np.inf, repr=False)
    high: float | np.ndarray = field(default=np.inf, repr=False)
    dtype: Any = field(default=np.float32, repr=False)

    def to_gym(self) -> gym.spaces.Box:
        return gym.spaces.Box(self.low, self.high, self.shape, self.dtype)

    def flatten_obs(self, observation: np.ndarray) -> np.ndarray:
        flat_obs = np.reshape(observation, -1)
        return flat_obs


@dataclass(frozen=True)
class CompositeSpace(FlattenableSpace[dict[Any, Any]]):
    space_dict: dict[Any, Any]

    def __getitem__(self, key: Any) -> Any:
        return self.space_dict[key]

    def to_gym(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {key: space.to_gym() for key, space in self.space_dict.items()}
        )

    def flatten_obs(self, observation: dict[Any, Any]) -> np.ndarray:
        sub_obs = [
            space.flatten_obs(observation[key])
            for key, space in self.space_dict.items()
        ]
        flat_obs = np.concatenate(sub_obs)
        return flat_obs


@dataclass(frozen=True, repr=False)
class MultiFiniteSpace(list, FlattenableSpace[T]):
    elements: InitVar[dict[Any, FiniteSpace]]
    reverse_map: dict[Any, Any] = field(init=False)

    def __post_init__(self, elements) -> None:
        super().__init__(list(chain(*elements.values())))
        reverse_map = {value: key for key, space in elements.items() for value in space}
        object.__setattr__(self, "reverse_map", reverse_map)

    def __repr__(self) -> str:
        return self.__class__.__qualname__ + f"({self.reverse_map})"

    def to_gym(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self))

    def sample(self, mask: Any = None) -> T:
        return random.choice(self)

    def flatten_obs(self, observation: T) -> np.ndarray:
        flat_obs = np.zeros(len(self))
        flat_obs[self.index(observation)] = 1
        return flat_obs
