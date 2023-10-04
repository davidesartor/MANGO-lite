from __future__ import annotations
from typing import Any, Generic, TypeVar
from utils.abstractionlayers import AbstractLayer, EnvLayer, MangoEnv, MangoLayer
from utils.buffers import AbstractionBuffer, BufferStopCondition

from utils.compatibilityfunctions import CompatibilityFunction
from utils.concepts import ConceptFunction, Identity
from utils.dynamicpolicies import DynamicPolicyFactory
from utils.intrinsicreward import StateTransitionRewardGenerator
from utils.replaymemory import ListReplayMemory

EnvObsType = TypeVar("EnvObsType")
EnvActType = TypeVar("EnvActType")
NewObsType = TypeVar("NewObsType")
NewActType = TypeVar("NewActType")


class Mango(Generic[EnvObsType, EnvActType]):
    def __init__(
        self,
        environment: EnvLayer[EnvObsType, EnvActType],
        concept_functions: list[ConceptFunction[EnvObsType, Any]],
        compatibility_functions: list[CompatibilityFunction[Any, Any]],
        dynamic_policy_factories: list[DynamicPolicyFactory[Any]],
        base_concept: ConceptFunction[EnvObsType, Any] | None = None,
    ) -> None:

        if base_concept is None:
            base_concept = Identity(environment.observation_space)
        concept_functions = [base_concept] + concept_functions

        if len(dynamic_policy_factories) == 1:
            dynamic_policy_factories *= len(compatibility_functions)

        # make buffers, start from the highest concept function towards the bottom
        self.buffers: list[AbstractionBuffer[EnvObsType, Any]] = []

        self.buffers.append(AbstractionBuffer(concept_function=concept_functions[-1]))
        for concept_function in reversed(concept_functions[:-1]):
            self.buffers.append(AbstractionBuffer(concept_function, self.buffers[-1]))
        self.buffers = self.buffers[::-1]

        # make layers, start from enviroment towards the top
        self.layers: list[AbstractLayer[Any, Any]] = []

        self.layers.append(MangoEnv(environment, self.buffers[0]))

        for buffer, compatibility, dynamic_policy_factory in zip(
            self.buffers[1:], compatibility_functions, dynamic_policy_factories
        ):
            layer = MangoLayer(
                lower_layer=self.layers[-1],
                observation_buffer=buffer,
                intra_layer_policy=dynamic_policy_factory(
                    comand_space=compatibility.action_space,
                    observation_space=self.layers[-1].observation_space,
                    action_space=self.layers[-1].action_space,
                ),
                stop_condition=BufferStopCondition(buffer),
                reward_generator=StateTransitionRewardGenerator(buffer, compatibility),
                replay_memory=ListReplayMemory(),
            )
            self.layers.append(layer)

    def reset(self) -> None:
        self.layers[-1].reset()
