from dataclasses import dataclass
from functools import cached_property
import random
from typing import Callable, Optional, TypeVar, Container, Any, Generic, Protocol
from utils.buffers import ObsBuffer, UpdatableObsBuffer
from utils.replaymemory import ReplayMemory
from utils.intrinsicreward import IntrinsicRewardGenerator
from utils.policies import Policy
from utils.dynamicpolicies import DynamicPolicy
from utils.spaces import SASRI, FlattenableSpace, SpaceType

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
LowObsType = TypeVar("LowObsType")
LowActType = TypeVar("LowActType")


class EnvLayer(Protocol[ObsType, ActType]):
    observation_space: SpaceType[ObsType]
    action_space: SpaceType[ActType]

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        ...

    def reset(self) -> tuple[ObsType, dict[str, Any]]:
        ...


class AbstractLayer(EnvLayer[ObsType, ActType], Protocol[ObsType, ActType]):
    @property
    def current_observation(self) -> ObsType:
        ...


@dataclass(frozen=True, eq=False)
class MangoEnv(AbstractLayer[ObsType, ActType], Generic[ObsType, ActType, LowObsType]):
    environment: EnvLayer[LowObsType, ActType]
    observation_buffer: UpdatableObsBuffer[LowObsType, ObsType]

    def __post_init__(self) -> None:
        self.reset()

    @property
    def current_observation(self) -> ObsType:
        return self.observation_buffer.current_observation

    @property
    def observation_space(self) -> SpaceType[ObsType]:
        return self.observation_buffer.observation_space

    @property
    def action_space(self) -> SpaceType[ActType]:
        return self.environment.action_space

    def step(self, action: ActType) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.environment.step(action)
        self.observation_buffer.push(state)
        return self.current_observation, reward, truncated, terminated, info

    def reset(self) -> tuple[ObsType, dict[str, Any]]:
        state, info = self.environment.reset()
        self.observation_buffer.reset(state)
        return self.current_observation, info


@dataclass(frozen=True, eq=False)
class MangoLayer(
    AbstractLayer[ObsType, ActType],
    Generic[ObsType, ActType, LowObsType, LowActType],
):
    lower_layer: AbstractLayer[LowObsType, LowActType]
    observation_buffer: ObsBuffer[ObsType]
    intra_layer_policy: DynamicPolicy[ActType, LowObsType, LowActType]
    stop_condition: Callable[[], bool]
    reward_generator: IntrinsicRewardGenerator[ActType, LowObsType, LowActType]
    replay_memory: ReplayMemory[SASRI[LowObsType, LowActType]]

    @property
    def current_observation(self) -> ObsType:
        return self.observation_buffer.current_observation

    @property
    def observation_space(self) -> SpaceType[ObsType]:
        return self.observation_buffer.observation_space
    
    @property
    def action_space(self) -> SpaceType[ActType]:
        return self.intra_layer_policy.comand_space

    def step(
        self, action: ActType, verbose: bool = False
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:

        low_sasri_list = self.iterate_policy(
            self.intra_layer_policy.get_policy(action), verbose=verbose
        )
        if verbose:
            print("iteration_length: ", len(low_sasri_list))
        for sasri in low_sasri_list:
            self.replay_memory.push(sasri)

        accumulated_reward = sum(float(sasri.reward) for sasri in low_sasri_list)
        info: dict[str, Any] = {}
        truncated = any(sasri.truncated for sasri in low_sasri_list)
        terminated = any(sasri.terminated for sasri in low_sasri_list)
        return self.current_observation, accumulated_reward, truncated, terminated, info

    def iterate_policy(
        self,
        policy: Policy[LowObsType, LowActType],
        verbose: bool = False,
        max_steps: int = 20,
    ) -> list[SASRI[LowObsType, LowActType]]:

        sasri_list: list[SASRI[LowObsType, LowActType]] = []
        for _ in range(max_steps):
            start_state = self.lower_layer.current_observation
            action = policy(start_state)
            end_state, reward, truncated, terminated, info = self.lower_layer.step(
                action
            )
            sasri = SASRI(
                start_state, action, end_state, reward, truncated, terminated, info
            )
            sasri_list.append(self.reward_generator.generate_info(sasri))
            if self.stop_condition():
                sasri = SASRI.from_sasri_template(sasri, truncated=True)
                break
            elif sasri.truncated or sasri.terminated:
                break

        if verbose:
            for sasri in sasri_list:
                print(f"ACTION {sasri.action} || ENV REWARD {sasri.reward}")
                intrinsic_rewards = {
                    comand: self.reward_generator.generate_reward(comand, sasri).reward
                    for comand in self.action_space # type: ignore
                }
                print(f"INTRINSIC REWARD {intrinsic_rewards}")
        return sasri_list

    def train(self, emphasis: Optional[Callable[[ActType], float]] = None) -> None:
        self.intra_layer_policy.train(
            sasri_list=self.replay_memory.sample(),
            mapper=self.reward_generator.generate_reward,
            emphasis=emphasis,
        )

    def set_exploration_rate(self, exploration_rate: float) -> None:
        self.intra_layer_policy.set_exploration_rate(exploration_rate)

    def reset(self) -> tuple[ObsType, dict[str, Any]]:
        low_state, info = self.lower_layer.reset()
        return self.current_observation, info
