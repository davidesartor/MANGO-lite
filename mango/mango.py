from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from typing import Any, Generic, NewType, Sequence, TypeVar
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from .concepts import ActionCompatibility, Concept, IdentityConcept

from .dynamicpolicies import DQnetPolicyMapper
from .environments import Environment
from .utils import ReplayMemory


ObsType = TypeVar("ObsType")


@dataclass(eq=False, slots=True)
class MangoLayer(Generic[ObsType]):
    concept: Concept[ObsType]
    action_compatibility: ActionCompatibility 
    lower_layer: MangoLayer[ObsType] = field(repr=False)

    replay_memory: ReplayMemory = field(default_factory=ReplayMemory, repr=False)
    policy: DQnetPolicyMapper = field(init=False)

    def __post_init__(self) -> None:
        self.policy = DQnetPolicyMapper(
            comand_space=self.concept.comand_space,  # type: ignore[protocol invariance]
            action_space=self.lower_layer.action_space,
        )

    @property
    def action_space(self) -> spaces.Discrete:
        return self.concept.action_space


@dataclass(eq=False, slots=True, init=False)
class Mango(Generic[ObsType]):
    layers: list[MangoLayer]

    def __init__(
        self,
        environment: Environment[ObsType, int],
        concepts: Sequence[ExtendedConcept[ObsType, npt.NDArray, int]],
        base_concept: Concept[ObsType, npt.NDArray] = IdentityConcept(),
    ) -> None:
        all_concepts = [base_concept] + list(concepts)

        self.layers = []
        for concept in self.concepts:
            self.layers.append(
                MangoLayer(
                    concept,
                    DQnetPolicyMapper(
                        comand_space=concept.comand_space,  # type: ignore
                        action_space=self.layers[-1].comand_space,
                    ),
                )
            )

        self.layers = [
            MangoLayer(concept, policy)
            for concept, policy in zip(self.concepts, self.intralayer_policies)
        ]

    @property
    def option_space(self) -> list[spaces.Discrete]:
        return [policy.comand_space for policy in self.intralayer_policies]

    def execute_option(self, action: int, layer: int = 0) -> None:
        ...

    def train(self, epochs: int, layer: int = -1) -> None:
        if layer == -1:
            layer = len(self.intralayer_policies) - 1

        for epoch in range(epochs):
            self.environment.reset()

            self.execute_option(
                action=int(self.option_space[layer].sample()), layer=layer
            )
            for layer in self.layers:
                layer.train(
                    transitions=self.environment.transitions,
                    reward_generator=policy.compatibility,
                )

    def reset(self) -> None:
        state, info = self.environment.reset()
        TODO


"""def iterate_policy(
    start_state: NamedMultiArray,
    policy: Policy,
    environment: AbstractEnvironment,
) -> Iterable[Transition]:
    action = policy(start_state)
    step = environment.step(action)
    yield Transition(start_state, action, *step)
    
    while not (step.terminated or step.truncated):
        start_state = step.next_state
        action = policy(start_state)
        step = environment.step(action)        
        yield Transition(start_state, action, *step)"""
