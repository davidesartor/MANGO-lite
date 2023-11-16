from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol, Sequence, Callable

from .policies import DQnetPolicy
from ..utils import Transition, ObsType, ActType
from .. import spaces


class DynamicPolicy(Protocol):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete

    def get_action(
        self, comand: ActType, obs: ObsType, randomness: float = 0.0
    ) -> ActType:
        ...

    def train(
        self,
        comand: ActType,
        transitions: Sequence[Transition],
        reward_generator: Callable[[ActType, ObsType, ObsType], float],
    ) -> float | None:
        ...


@dataclass(eq=False, slots=True, repr=True)
class DQnetPolicyMapper(DynamicPolicy):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete
    policy_params: InitVar[dict[str, Any]] = dict()
    policies: dict[ActType, DQnetPolicy] = field(init=False, repr=False)

    def __post_init__(self, policy_params):
        self.policies = {
            comand: DQnetPolicy(self.action_space, **policy_params)
            for comand in self.comand_space
        }

    def get_action(self, comand: ActType, obs: ObsType, randomness: float = 0.0):
        return self.policies[comand].get_action(obs, randomness)

    def train(
        self,
        comand: ActType,
        transitions: Sequence[Transition],
        reward_generator: Callable[[ActType, ObsType, ObsType], float],
    ) -> float | None:
        training_transitions = []
        for transition in transitions:
            new_reward = reward_generator(
                comand, transition.start_obs, transition.next_obs
            )
            training_transitions.append(transition._replace(reward=new_reward))
        return self.policies[comand].train(transitions=training_transitions)
