from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from typing import Any, Generic, Iterator, Optional, Sequence, TypeVar
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from .concepts import ActionCompatibility, Concept, IdentityConcept

from .dynamicpolicies import DQnetPolicyMapper
from .environments import Environment
from .utils import ReplayMemory, Transition


ObsType = TypeVar("ObsType")


@dataclass(eq=False, slots=True)
class MangoEnv(Generic[ObsType]):
    concept: Concept[ObsType]
    environment: Environment[ObsType]

    @property
    def action_space(self) -> spaces.Discrete:
        return self.environment.action_space

    def step(
        self, action: int, env_state: ObsType
    ) -> tuple[ObsType, float, bool, bool, dict]:
        return self.environment.step(action)


@dataclass(eq=False, slots=True)
class MangoLayer(Generic[ObsType]):
    concept: Concept[ObsType]
    action_compatibility: ActionCompatibility

    lower_layer: MangoLayer[ObsType] | MangoEnv[ObsType] = field(repr=False)
    current_state: ObsType = field(init=False, repr=False)

    replay_memory: ReplayMemory = field(default_factory=ReplayMemory, repr=False)
    policy: DQnetPolicyMapper = field(init=False)

    def __post_init__(self) -> None:
        self.policy = DQnetPolicyMapper(
            comand_space=self.action_compatibility.action_space,
            action_space=self.lower_layer.action_space,
        )

    @property
    def action_space(self) -> spaces.Discrete:
        return self.action_compatibility.action_space

    def step(
        self, action: int, env_state: ObsType
    ) -> tuple[ObsType, float, bool, bool, dict]:
        transitions = []
        while not terminated_cond:
            pass

    def iterate_policy(
        self, action: int, env_state: ObsType
    ) -> Iterator[tuple[Transition,Transition]]:
        start_state_lower = self.lower_layer.concept.abstract(env_state)
        start_state_upper = self.concept.abstract(env_state)

        lower_action = self.policy.get_action(comand=action, state=start_state_lower)
        env_state, reward, terminated, truncated, info = self.lower_layer.step(
            action, env_state
        )

        next_state_lower = self.lower_layer.concept.abstract(env_state)
        next_state_upper = self.concept.abstract(env_state)

    def reset(self, *args, **kwargs) -> None:
        self.current_state, info = self.lower_layer.reset(*args, **kwargs)


@dataclass(eq=False, slots=True, init=False)
class Mango(Generic[ObsType]):
    environment: Environment[ObsType]
    layers: list[MangoLayer[ObsType] | MangoEnv[ObsType]]

    def __init__(
        self,
        environment: Environment[ObsType],
        concepts: Sequence[Concept[ObsType]],
        action_compatibilities: Sequence[ActionCompatibility],
        base_concept: Concept[ObsType] = IdentityConcept(),
    ) -> None:
        self.environment = environment

        self.layers = [MangoEnv(base_concept, self.environment)]
        for concept, compatibility in zip(concepts, action_compatibilities):
            self.layers.append(MangoLayer(concept, compatibility, self.layers[-1]))

    @property
    def option_space(self) -> list[spaces.Discrete]:
        return [layer.action_space for layer in self.layers]

    def execute_option(self, action: int, layer: int = 0) -> None:
        ...

    def train(self, epochs: int, layer_idx: int = -1) -> None:
        if layer_idx == -1:
            layer_idx = len(self.layers) - 1

        for epoch in range(epochs):
            self.environment.reset()

            self.execute_option(
                action=int(self.option_space[layer_idx].sample()), layer=layer_idx
            )
            for layer in self.layers:
                layer.train()

    def reset(self) -> None:
        state, info = self.environment.reset()
        TODO
