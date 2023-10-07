from dataclasses import InitVar, dataclass, field, replace
from typing import Any, Protocol, Sequence, TypeVar, Callable, Iterable, Optional

import numpy.typing as npt
from gymnasium import spaces

from .concepts import ActionCompatibility
from .policies import Policy, DQnetPolicy
from .utils import Transition


class DynamicPolicy(Protocol):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete

    def get_action(self, comand: int, state: npt.NDArray) -> int:
        ...

    def train(
        self,
        transitions: Sequence[tuple[Transition, Transition]],
        reward_generator: ActionCompatibility,
        emphasis: Callable[[int], float] = lambda _: 1.0,
    ) -> None:
        ...

    def set_exploration_rate(self, exploration_rate: float) -> None:
        ...


@dataclass(eq=False, slots=True)
class DQnetPolicyMapper(DynamicPolicy):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete

    exploration_rate: float = field(init=False, default=1.0, repr=False)
    policies: dict[int, Policy] = field(init=False, repr=False)

    def __post_init__(self):
        self.policies = {
            comand: DQnetPolicy(action_space=self.action_space)
            for comand in range(int(self.comand_space.n))
        }

    def get_action(self, comand: int, state: npt.NDArray) -> int:
        return self.policies[comand].get_action(state)

    def train(
        self,
        transitions: Sequence[tuple[Transition, Transition]],
        reward_gen: ActionCompatibility,
        emphasis: Callable[[int], float] = lambda _: 1.0,
    ) -> None:
        emph_tot = sum([emphasis(comand) for comand in range(self.comand_space.n)])
        for comand, policy in self.policies.items():
            training_transitions = [
                t_lower._replace(
                    reward=reward_gen(comand, t_upper.start_state, t_upper.next_state)
                )
                for t_lower, t_upper in transitions
            ]

            for cycle in range(int(emphasis(comand) / emph_tot * self.comand_space.n)):
                policy.train(lower_transitions)  # type: ignore[need correct typehinting of transition]

    def set_exploration_rate(self, exploration_rate: float) -> None:
        for comand, policy in self.policies.items():
            policy.set_exploration_rate(exploration_rate)