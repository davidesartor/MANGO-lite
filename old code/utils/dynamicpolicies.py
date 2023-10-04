from dataclasses import InitVar, dataclass, field
from typing import TypeVar, Callable, Iterable, Optional, Protocol

from utils.policies import Policy, PolicyFactory
from utils.spaces import SASRI, SpaceType, FiniteSpace

# region protocols

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
ComType = TypeVar("ComType")


class DynamicPolicy(Protocol[ComType, ObsType, ActType]):
    observation_space: SpaceType[ObsType]
    action_space: SpaceType[ActType]
    comand_space: SpaceType[ComType]

    def __call__(self, comand: ComType, state: ObsType) -> ActType:
        return self.get_action(comand, state)

    def get_action(self, comand: ComType, state: ObsType) -> ActType:
        return self.get_policy(comand).get_action(state)

    def get_policy(self, comand: ComType) -> Policy[ObsType, ActType]:
        ...

    def set_exploration_rate(self, exploration_rate: float) -> None:
        ...

    def train(
        self,
        sasri_list: Iterable[SASRI[ObsType, ActType]],
        mapper: Callable[[ComType, SASRI[ObsType, ActType]], SASRI[ObsType, ActType]],
        emphasis: Optional[Callable[[ComType], float]] = None,
    ) -> None:
        ...


DynPolicyType_co = TypeVar("DynPolicyType_co", bound=DynamicPolicy, covariant=True)


class DynamicPolicyFactory(Protocol[DynPolicyType_co]):
    def __call__(
        self,
        comand_space: SpaceType,
        observation_space: SpaceType,
        action_space: SpaceType,
    ) -> DynPolicyType_co:
        return self.make(comand_space, observation_space, action_space)

    def make(
        self,
        comand_space: SpaceType,
        observation_space: SpaceType,
        action_space: SpaceType,
    ) -> DynPolicyType_co:
        ...


# endregion


# region discrete policy mapper


@dataclass(frozen=True, eq=False)
class DiscretePolicyMapper(DynamicPolicy[ComType, ObsType, ActType]):
    comand_space: FiniteSpace[ComType]
    observation_space: SpaceType[ObsType]
    action_space: SpaceType[ActType]
    policy_factory: InitVar[PolicyFactory]
    cycles_per_train: int = 0
    policy_map: dict[ComType, Policy[ObsType, ActType]] = field(init=False)

    def __post_init__(self, policy_factory: PolicyFactory) -> None:
        if not self.cycles_per_train:
            object.__setattr__(self, "cycles_per_train", len(self.comand_space))
        policy_map = {
            comand: policy_factory(self.observation_space, self.action_space)
            for comand in self.comand_space
        }
        object.__setattr__(self, "policy_map", policy_map)

    def get_policy(self, comand: ComType) -> Policy[ObsType, ActType]:
        return self.policy_map[comand]

    def set_exploration_rate(self, exploration_rate: float) -> None:
        for comand, policy in self.policy_map.items():
            policy.set_exploration_rate(exploration_rate)

    def train(
        self,
        sasri_list: Iterable[SASRI[ObsType, ActType]],
        mapper: Callable[[ComType, SASRI[ObsType, ActType]], SASRI[ObsType, ActType]],
        emphasis: Optional[Callable[[ComType], float]] = None,
    ) -> None:
        if emphasis is None:
            emphasis = lambda _: 1 / self.cycles_per_train

        for comand, policy in self.policy_map.items():
            for cycle in range(int(emphasis(comand) * self.cycles_per_train)):
                policy.train(mapper(comand, sasri) for sasri in sasri_list)


@dataclass(frozen=True, eq=False)
class DiscretePolicyMapperFactory(DynamicPolicyFactory[DiscretePolicyMapper]):
    policy_factory: PolicyFactory

    def make(
        self,
        comand_space: FiniteSpace,
        observation_space: SpaceType,
        action_space: FiniteSpace,
    ) -> DiscretePolicyMapper:
        return DiscretePolicyMapper(
            comand_space=comand_space,
            observation_space=observation_space,
            action_space=action_space,
            policy_factory=self.policy_factory,
        )


# endregion
