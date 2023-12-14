from dataclasses import InitVar, dataclass, field
from typing import Any

from mango import spaces
from mango.protocols import ActType, ObsType, TrainInfo, Transition
from mango.protocols import DynamicPolicy, Policy


@dataclass(eq=False, slots=True, repr=True)
class PolicyMapper(DynamicPolicy):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete
    policy_cls: InitVar[type[Policy]]
    policy_params: InitVar[dict[str, Any]]

    policies: dict[ActType, Policy] = field(init=False, repr=False)

    def __post_init__(self, policy_cls: type[Policy], policy_params: dict[str, Any]):
        self.policies = {
            comand: policy_cls.make(self.action_space, **policy_params)
            for comand in self.comand_space
        }

    def get_action(self, comand: ActType, obs: ObsType, randomness: float = 0.0):
        return self.policies[comand].get_action(obs, randomness)

    def train(self, comand: ActType, transitions: list[Transition]) -> TrainInfo:
        return self.policies[comand].train(transitions)

    @classmethod
    def make(
        cls, comand_space: spaces.Discrete, action_space: spaces.Discrete, **kwargs
    ) -> DynamicPolicy:
        return cls(comand_space, action_space, **kwargs)
