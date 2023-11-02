from dataclasses import dataclass, field
from typing import Protocol, Sequence, Callable

import numpy.typing as npt
import gymnasium as gym

from .actions import ActionCompatibility
from .policies import Policy, DQnetPolicy
from .utils import Transition, torch_style_repr


class DynamicPolicy(Protocol):
    comand_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete

    def get_action(self, comand: int, state: npt.NDArray) -> int:
        ...

    def train(
        self,
        transitions: Sequence[tuple[Transition, Transition]],
        reward_generator: ActionCompatibility,
    ) -> None:
        ...


@dataclass(eq=False, slots=True, repr=False)
class DQnetPolicyMapper(DynamicPolicy):
    comand_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete

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
    ) -> None:
        for comand, policy in self.policies.items():
            training_transitions = [
                t_lower._replace(
                    reward=reward_gen(comand, t_upper.start_state, t_upper.next_state)
                )
                for t_lower, t_upper in transitions
            ]
            policy.train(training_transitions)

    def __repr__(self) -> str:
        params = {f"{comand}": str(policy) for comand, policy in self.policies.items()}
        return torch_style_repr(self.__class__.__name__, params)
