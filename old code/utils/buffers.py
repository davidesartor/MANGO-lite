from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Generic, Iterable, Protocol, TypeVar

import numpy as np

from utils.spaces import SpaceType, CompositeSpace
from utils.concepts import ConceptFunction

InObsType = TypeVar("InObsType")
OutObsType = TypeVar("OutObsType")


class ObsBuffer(Protocol[OutObsType]):
    observation_space: SpaceType[OutObsType]
    current_transition: tuple[OutObsType, OutObsType] | None
    current_observation: OutObsType


class UpdatableObsBuffer(ObsBuffer[OutObsType], Protocol[InObsType, OutObsType]):
    input_space: SpaceType[InObsType]

    def push(self, input_observation: InObsType) -> None:
        ...

    def reset(self, input_observation: InObsType) -> None:
        ...


def obs_not_equal(obs1: OutObsType, obs2: OutObsType) -> bool:
    if isinstance(obs1, np.ndarray) and isinstance(obs2, np.ndarray):
        return not np.array_equal(obs1, obs2)
    if isinstance(obs1, dict) and isinstance(obs2, dict):
        try:
            return not any(obs_not_equal(obs1[key], obs2[key]) for key in obs1.keys())
        except KeyError:
            raise ValueError("Observations must have the same keys")
    return obs1 != obs2


@dataclass(frozen=True, eq=False)
class ConceptBuffer(UpdatableObsBuffer[InObsType, OutObsType]):
    concept_function: ConceptFunction[InObsType, OutObsType]
    current_observation: OutObsType = field(init=False, repr=False)
    previous_observation: OutObsType | None = field(init=False, repr=False)

    input_space: SpaceType[InObsType] = field(init=False, repr=False)
    observation_space: SpaceType[OutObsType] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        input_space = self.concept_function.input_observation_space
        observation_space = self.concept_function.output_observation_space
        object.__setattr__(self, "input_space", input_space)
        object.__setattr__(self, "observation_space", observation_space)

    @property
    def current_transition(self) -> tuple[OutObsType, OutObsType] | None:
        if self.previous_observation is None or self.current_observation is None:
            return None
        if obs_not_equal(self.previous_observation, self.current_observation):
            return None
        return self.previous_observation, self.current_observation

    def push(self, input_observation: InObsType) -> None:
        object.__setattr__(self, "previous_observation", self.current_observation)
        object.__setattr__(
            self, "current_observation", self.concept_function(input_observation)
        )

    def reset(self, input_observation: InObsType) -> None:
        self.push(input_observation)
        object.__setattr__(self, "previous_observation", None)


UpperObsType = TypeVar("UpperObsType")


@dataclass(frozen=True, eq=False)
class AbstractionBuffer(
    UpdatableObsBuffer[InObsType, dict[str, Any]],
    Generic[InObsType, OutObsType],
):
    concept_function: ConceptFunction[InObsType, OutObsType]
    current_observation: OutObsType = field(init=False, repr=False)
    previous_observation: OutObsType | None = field(init=False, repr=False)
    upper_buffer: AbstractionBuffer[InObsType, dict[str, Any]] | None = field(
        default=None, repr=False
    )

    input_space: SpaceType[InObsType] = field(init=False, repr=False)
    observation_space: CompositeSpace = field(init=False, repr=False)

    def __post_init__(self) -> None:
        input_space = self.concept_function.input_observation_space

        concepts = {
            self.concept_function.name: self.concept_function.output_observation_space
        }
        if self.upper_buffer is not None:
            concepts = {**concepts, **self.upper_buffer.observation_space.space_dict}
        observation_space = CompositeSpace(concepts)

        object.__setattr__(self, "input_space", input_space)
        object.__setattr__(self, "observation_space", observation_space)

    @property
    def current_transition(self) -> tuple[OutObsType, OutObsType] | None:
        if self.previous_observation is None or self.current_observation is None:
            return None
        if obs_not_equal(self.previous_observation, self.current_observation):
            return None
        return self.previous_observation, self.current_observation

    def push(self, input_observation: InObsType) -> None:
        if self.upper_buffer is not None:
            self.upper_buffer.push(input_observation)

        object.__setattr__(self, "previous_observation", self.current_observation)
        concepts = {
            self.concept_function.name: self.concept_function(input_observation)
        }
        if self.upper_buffer is not None:
            concepts = {**concepts, **self.upper_buffer.current_observation}

        object.__setattr__(self, "current_observation", concepts)

    def reset(self, input_observation: InObsType) -> None:
        concepts = {
            self.concept_function.name: self.concept_function(input_observation)
        }
        if self.upper_buffer is not None:
            self.upper_buffer.reset(input_observation)
            concepts = {**concepts, **self.upper_buffer.current_observation}

        object.__setattr__(self, "current_observation", concepts)
        object.__setattr__(self, "previous_observation", None)


@dataclass(frozen=True, eq=False)
class BufferStopCondition:
    buffer: ObsBuffer[Any]

    def __call__(self) -> bool:
        return self.buffer.current_transition is not None
