from dataclasses import InitVar, dataclass, field, replace
from typing import Any, Protocol, Sequence, TypeVar, Callable, Iterable, Optional

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from .policies import Policy, DQnetPolicy
from .utils import Transition


ObsType = TypeVar("ObsType")
AbsObsType = TypeVar("AbsObsType")
ActType = TypeVar("ActType")
AbsActType = TypeVar("AbsActType")


class DynamicPolicy(Protocol[AbsActType, ObsType, ActType]):
    comand_space: spaces.Space[AbsActType]
    action_space: spaces.Space[ActType]
    exploration_rate: float = 1.0

    def get_action(self, comand: AbsActType, state: ObsType) -> ActType:
        ...

    def train(
        self,
        transitions: Sequence[Transition[tuple[ObsType, AbsObsType], ActType]],
        reward_generator: Callable[[AbsActType, AbsObsType, AbsObsType], float],
        emphasis: Callable[[AbsActType], float] = lambda _: 1.0,
    ) -> None:
        ...

    def set_exploration_rate(self, exploration_rate: float) -> None:
        ...


@dataclass(eq=False, repr=False, slots=True)
class DQnetPolicyMapper(DynamicPolicy[int, npt.NDArray, int]):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete

    exploration_rate: float = field(init=False, default=1.0)
    policies: dict[int, Policy[npt.NDArray, int]] = field(init=False)

    def __post_init__(self):
        self.policies = {
            comand: DQnetPolicy(action_space=self.action_space)
            for comand in range(int(self.comand_space.n))
        }

    def get_action(self, comand: int, state: npt.NDArray) -> int:
        return self.policies[comand].get_action(state)

    def train(
        self,
        transitions: Sequence[Transition[tuple[npt.NDArray, npt.NDArray], int]],
        reward_generator: Callable[[int, npt.NDArray, npt.NDArray], float],
        emphasis: Callable[[int], float] = lambda _: 1.0,
    ) -> None:
        emph_tot = sum([emphasis(comand) for comand in range(self.comand_space.n)])
        for comand, policy in self.policies.items():
            lower_transitions = [
                (s1, act, s2, reward_generator(comand, as1, as2), ter, tru, info)
                for (s1, as1), act, (s2, as2), rew, ter, tru, info in transitions
            ]
            for cycle in range(int(emphasis(comand) / emph_tot * self.comand_space.n)):
                policy.train(lower_transitions)  # type: ignore[need correct typehinting of transition]

    def set_exploration_rate(self, exploration_rate: float) -> None:
        for comand, policy in self.policies.items():
            policy.set_exploration_rate(exploration_rate)
