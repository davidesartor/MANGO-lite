from dataclasses import dataclass, field
from typing import Protocol, Sequence, Callable, TypeVar

import numpy.typing as npt
import gymnasium as gym

from .policies import Policy, DQnetPolicy
from ..utils import Transition, torch_style_repr

ObsType = TypeVar("ObsType", bound=npt.NDArray)

class DynamicPolicy(Protocol[ObsType]):
    comand_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete

    def get_action(
        self, comand: int, state: ObsType, randomness: float = 0.0
    ) -> int:
        ...

    def train(
        self,
        comand: int,
        transitions: Sequence[Transition],
        reward_generator: Callable[[int, ObsType, ObsType], float]
    ) -> float:
        ...


@dataclass(eq=False, slots=True)
class DQnetPolicyMapper(DynamicPolicy):
    comand_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete
    policies: dict[int, Policy] = field(init=False)

    def __post_init__(self):
        self.policies = {
            comand: DQnetPolicy(action_space=self.action_space)
            for comand in range(int(self.comand_space.n))
        }
        self.loss_log = tuple([] for _ in range(self.action_space.n))

    def get_action(self, comand: int, state: npt.NDArray, randomness: float = 0.0):
        return self.policies[comand].get_action(state, randomness)

    def train(
        self,
        comand: int,
        transitions: Sequence[Transition],
        reward_generator: ActionCompatibility,
    ) -> float | None:
        training_transitions = []
        for transition in transitions:
            new_reward = reward_generator(comand, transition.start_state, transition.next_state)
            training_transitions.append(transition._replace(reward=new_reward))
        return self.policies[comand].train(transitions=training_transitions)
