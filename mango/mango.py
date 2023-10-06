from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Generic, NewType, TypeVar
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from .concepts import Concept, ExtendedConcept, IdentityConcept

from .dynamicpolicies import DQnetPolicyMapper
from .environments import Environment


ObsType = TypeVar("ObsType")


@dataclass(eq=False, slots=True)
class Mango(Generic[ObsType]):
    environment: Environment[ObsType, int] = field(repr=False)
    concepts: list[ExtendedConcept[ObsType, npt.NDArray, int]]
    base_concept: Concept[ObsType, npt.NDArray] = IdentityConcept()

    intralayer_policies: list[DQnetPolicyMapper] = field(init=False)

    def __post_init__(self):
        self.intralayer_policies = [
            DQnetPolicyMapper(
                comand_space=self.concepts[0].comand_space,  # type: ignore
                action_space=self.environment.action_space,  # type: ignore
            )
        ]

        for concept in self.concepts:
            self.intralayer_policies.append(
                DQnetPolicyMapper(
                    comand_space=concept.comand_space,  # type: ignore
                    action_space=self.intralayer_policies[-1].comand_space,
                )
            )

    @property
    def option_space(self) -> list[spaces.Discrete]:
        return [policy.comand_space for policy in self.intralayer_policies]

    def execute_option(self, action: int, layer: int = 0) -> None:
        ...

    def train(self, epochs: int, from_layer: int = -1) -> None:
        ...

    def reset(self) -> None:
        ...
