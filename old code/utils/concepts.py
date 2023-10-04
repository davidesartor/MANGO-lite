from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar
from utils.spaces import SpaceType, CompositeSpace, FlattenableSpace
import numpy as np


InObsType = TypeVar("InObsType")
OutObsType = TypeVar("OutObsType")


class ConceptFunction(Protocol[InObsType, OutObsType]):
    input_observation_space: SpaceType[InObsType]
    output_observation_space: SpaceType[OutObsType]

    def __call__(self, input_state: InObsType | None) -> OutObsType | None:
        if input_state is None:
            return None
        return self.abstract(input_state)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def abstract(self, input_state: InObsType) -> OutObsType:
        ...


@dataclass(frozen=True, eq=False)
class Identity(ConceptFunction[InObsType, InObsType]):
    input_observation_space: SpaceType[InObsType]
    output_observation_space: SpaceType[InObsType] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self, "output_observation_space", self.input_observation_space
        )

    def abstract(self, input_state: InObsType) -> InObsType:
        return input_state


@dataclass(frozen=True, eq=False)
class StripFieldConcept(ConceptFunction[dict[str, SpaceType], Any]):
    field_name: str
    input_observation_space: CompositeSpace = field(repr=False)
    output_observation_space: SpaceType[Any] = field(init=False)

    def __post_init__(self):
        assert self.field_name in self.input_observation_space.space_dict
        out_obs_space = self.input_observation_space.space_dict[self.field_name]
        object.__setattr__(self, "output_observation_space", out_obs_space)

    @property
    def name(self) -> str:
        return f"Strip[{self.field_name}]Concept"

    def abstract(self, input_state: dict[str, Any]) -> Any:
        return input_state[self.field_name]


@dataclass(frozen=True, eq=False)
class ConditionalStripFieldConcept(ConceptFunction[dict[str, SpaceType], Any]):
    field_name: str
    condition: str
    input_observation_space: CompositeSpace = field(repr=False)
    output_observation_space: SpaceType[Any] = field(init=False)

    def __post_init__(self):
        assert self.field_name in self.input_observation_space.space_dict
        obs_space = self.input_observation_space.space_dict[self.field_name]
        assert isinstance(obs_space, CompositeSpace)
        object.__setattr__(
            self, "output_observation_space", next(iter(obs_space.space_dict.values()))
        )

    @property
    def name(self) -> str:
        return f"Strip[{self.field_name}][{self.condition}]Concept"

    def abstract(self, input_state: dict[str, Any]) -> Any:
        return input_state[self.field_name][input_state[self.condition]]
