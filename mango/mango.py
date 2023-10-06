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


@dataclass(eq=False, slots=True, repr=False)
class Mango(Generic[ObsType]):
    environment: Environment[ObsType, int]
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

    def __repr__(self) -> str:
        representation = f"Mango(\n"
        representation += f"    (env): {self.environment},\n"
        representation += f"    (options): \n"
        for i, option in enumerate(self.option_space):
            representation += f"        ({i}): {option},\n"
        representation += f"    (concepts): \n"
        representation += f"        (base): {self.base_concept},\n"
        for i, concept in enumerate(self.concepts):
            representation += f"        ({i+1}): {concept},\n"
        representation += f"    (policies): \n"
        for i, policy in enumerate(self.intralayer_policies):
            representation += f"        ({i}): {policy},\n"
        representation += f")"
        return representation
