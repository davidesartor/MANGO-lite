from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol, Sequence, TypeVar, Callable, Iterable, Optional

import numpy as np
from gymnasium import spaces

from .policies import Policy, DQnetPolicy
from .utils import Transition


ObsType = TypeVar("ObsType")
AbsObsType = TypeVar("AbsObsType")
ActType = TypeVar("ActType")
AbsActType = TypeVar("AbsActType")


class DynamicPolicy(Policy[ObsType, ActType], Protocol[AbsActType, ObsType, ActType]):
    @property
    def command_space(self) -> spaces.Space[AbsActType]:
        ...

    def get_action(self, comand: AbsActType, state: ObsType) -> ActType:
        ...

    def train(
        self,
        transitions: Sequence[Transition[tuple[ObsType, AbsObsType], ActType]],
        reward_generator: ActionCompatibility[AbsObsType, AbsActType],
        emphasis: Callable[[AbsActType], float] = lambda _: 1.0,
    ) -> None:
        ...



Array3d = np.ndarray[tuple[int, int, int], Any]

@dataclass(eq=False)
class DQnetPolicyMapper(DynamicPolicy[int, Array3d, int]):
    comand_space: spaces.Discrete
    observation_space: spaces.Box
    action_space: spaces.Discrete

    policies: list[Policy[Array3d, int]] = field(init=False)

    def __post_init__(self):
        self.policies = [
            DQnetPolicy(self.observation_space, self.action_space)
            for _ in range(int(self.comand_space.n))
        ]

    def set_exploration_rate(self, exploration_rate: float) -> None:
        for policy in self.policies:
            policy.set_exploration_rate(exploration_rate)

    def get_action(self, comand: int, state: Array3d) -> int:
        return self.policies[comand].get_action(state)

    def train(
        self,
        transitions: Sequence[Transition[Array3d, int]],
        reward_generator: ActionCompatibility[int, Array3d, int],
        emphasis: Callable[[int], float] = lambda _: 1.0,
    ) -> None:
        emph_tot = sum([emphasis(comand) for comand in range(self.comand_space.n)])
        for comand, policy in enumerate(self.policies):
            for cycle in range(int(emphasis(comand) / emph_tot * self.comand_space.n)):
                policy.train(
                    [t.replace(reward=reward_generator(comand, t)) for t in transitions]
                )
