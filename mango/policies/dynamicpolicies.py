from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol

from .. import spaces
from ..utils import ObsType, ActType
from .experiencereplay import TensorTransitionLists
from .policies import DQnetPolicy


class DynamicPolicy(Protocol):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete

    def get_action(self, comand: ActType, obs: ObsType, randomness: float = 0.0) -> ActType:
        ...

    def train(
        self,
        comand: ActType,
        transitions: TensorTransitionLists,
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
            comand: DQnetPolicy(self.action_space, **policy_params) for comand in self.comand_space
        }

    def get_action(self, comand: ActType, obs: ObsType, randomness: float = 0.0):
        return self.policies[comand].get_action(obs, randomness)

    def train(self, comand: ActType, transitions: TensorTransitionLists) -> float | None:
        return self.policies[comand].train(transitions)
