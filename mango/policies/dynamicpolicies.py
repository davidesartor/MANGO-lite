from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol, Sequence, Callable

from .policies import DQnetPolicy
from ..utils import Transition, ObsType, ActType
from ..actions.abstract_actions import AbstractActions
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
        abs_actions: AbstractActions,
    ) -> float | None:
        ...


@dataclass(eq=False, slots=True, repr=True)
class DQnetPolicyMapper(DynamicPolicy):
    comand_space: spaces.Discrete
    action_space: spaces.Discrete
    policy_params: InitVar[dict[str, Any]] = dict()
    obs_transform: Callable[[ObsType], ObsType] = field(default=lambda x: x, repr=False)
    policies: dict[ActType, DQnetPolicy] = field(init=False, repr=False)

    def __post_init__(self, policy_params):
        self.policies = {
            comand: DQnetPolicy(self.action_space, **policy_params)
            for comand in self.comand_space
        }

    def get_action(self, comand: ActType, obs: ObsType, randomness: float = 0.0):
        obs = self.obs_transform(obs)
        return self.policies[comand].get_action(obs, randomness)

    def train(
        self,
        comand: ActType,
        transitions: Sequence[Transition],
        reward_generator: Callable[[ActType, ObsType, ObsType], float],
    ) -> float | None:
        training_transitions = map(
            lambda t: t._replace(
                start_obs=self.obs_transform(t.start_obs),
                next_obs=self.obs_transform(t.next_obs),
                reward=reward_generator(comand, t.start_obs, t.next_obs),
            ),
            transitions,
        )
        return self.policies[comand].train(transitions=list(training_transitions))
