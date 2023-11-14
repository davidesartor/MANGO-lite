from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Generic, Iterator, Optional, Sequence, SupportsFloat, TypeVar
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .actions import ActionCompatibility
from .concepts import Concept, IdentityConcept
from .dynamicpolicies import DQnetPolicyMapper, DynamicPolicy
from .utils import ReplayMemory, Transition, torch_style_repr


ObsType = TypeVar("ObsType")


@dataclass(eq=False, slots=True, repr=False)
class MangoEnv(Generic[ObsType]):
    concept: Concept[ObsType]
    environment: gym.Env[ObsType, int]
    abs_state: npt.NDArray = field(init=False)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.environment.action_space  # type: ignore

    def step(
        self, action: int, randomness: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        if np.random.rand() < randomness:
            action = int(self.action_space.sample())
        env_state, reward, term, trunc, info = self.environment.step(action)
        self.abs_state = self.concept.abstract(env_state)
        info["mango:trajectory"] = [env_state]
        return env_state, reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        env_state, info = self.environment.reset(seed=seed, options=options)
        self.abs_state = self.concept.abstract(env_state)
        return env_state, info

    def __repr__(self) -> str:
        return torch_style_repr(
            self.__class__.__name__,
            {"concept": str(self.concept), "environment": str(self.environment)},
        )


@dataclass(eq=False, slots=True, repr=False)
class MangoLayer(Generic[ObsType]):
    concept: Concept[ObsType]
    action_compatibility: ActionCompatibility
    lower_layer: MangoLayer[ObsType] | MangoEnv[ObsType]
    replay_memory: ReplayMemory[tuple[Transition, npt.NDArray, npt.NDArray]] = field(
        init=False, default_factory=ReplayMemory
    )
    policy: DynamicPolicy = field(init=False)
    abs_state: npt.NDArray = field(init=False)
    intrinsic_reward_log: tuple[list[float], ...] = field(init=False)

    def __post_init__(self) -> None:
        self.policy = DQnetPolicyMapper(
            comand_space=self.action_compatibility.action_space,
            action_space=self.lower_layer.action_space,
        )
        self.intrinsic_reward_log = tuple([] for _ in range(self.action_space.n))

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.action_compatibility.action_space

    def step(
        self, action: int, randomness: float = 0.0, p_term: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        states_low = [self.lower_layer.abs_state]
        states_up = [self.abs_state]
        env_state_trajectory: list[ObsType] = []
        transitions_low: list[Transition] = []

        while True:
            low_action = self.policy.get_action(
                comand=action, state=states_low[-1], randomness=randomness
            )
            env_state, reward, term, trunc, info = self.lower_layer.step(
                low_action, randomness=randomness
            )
            env_state_trajectory.extend(info["mango:trajectory"])
            self.abs_state = self.concept.abstract(env_state)

            states_low.append(self.lower_layer.abs_state)
            states_up.append(self.abs_state)
            transition_low = Transition(
                states_low[-2], low_action, states_low[-1], reward, term, trunc, info
            )
            transitions_low.append(transition_low)

            if term or trunc or not np.all(states_up[-1] == states_up[-2]):
                break
            if random.random() < p_term:
                break

        self.replay_memory.extend(zip(transitions_low, states_up[:-1], states_up[1:]))
        accumulated_reward = sum(float(trans.reward) for trans in transitions_low)
        infos = {k: v for trans in transitions_low for k, v in trans.info.items()}
        infos["mango:trajectory"] = env_state_trajectory

        self.intrinsic_reward_log[action].append(
            self.action_compatibility(action, states_up[0], states_up[-1])
        )
        return env_state, accumulated_reward, term, trunc, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        env_state, info = self.lower_layer.reset(seed=seed, options=options)
        self.abs_state = self.concept.abstract(env_state)
        return env_state, info

    def __repr__(self) -> str:
        params = {
            "concept": str(self.concept),
            "act_comp": str(self.action_compatibility),
            "policy": str(self.policy),
        }
        return torch_style_repr(self.__class__.__name__, params)


class Mango(Generic[ObsType]):
    def __init__(
        self,
        environment: gym.Env[ObsType, int],
        concepts: Sequence[Concept[ObsType]],
        action_compatibilities: Sequence[ActionCompatibility],
        base_concept: Concept[ObsType] = IdentityConcept(),
    ) -> None:
        self.environment = MangoEnv(base_concept, environment)
        self.abstract_layers: list[MangoLayer[ObsType]] = []
        for concept, compatibility in zip(concepts, action_compatibilities):
            self.abstract_layers.append(
                MangoLayer(concept, compatibility, self.layers[-1])
            )
        self.reset()

    @property
    def layers(self) -> tuple[MangoEnv[ObsType] | MangoLayer[ObsType], ...]:
        return (self.environment, *self.abstract_layers)

    @property
    def option_space(self) -> tuple[gym.spaces.Discrete, ...]:
        return tuple(layer.action_space for layer in self.layers)

    def execute_option(
        self, layer_idx: int, action: int, randomness: float = 0.0, p_term: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        if layer_idx == 0:
            return self.environment.step(action, randomness)
        return self.abstract_layers[layer_idx - 1].step(
            action, randomness=randomness, p_term=p_term
        )

    def train(self, layer_idx: int, action: int) -> float | None:
        if layer_idx == 0:
            raise ValueError("Cannot train base layer")

        layer = self.abstract_layers[layer_idx - 1]
        loss = layer.policy.train(
            comand=layer_idx,
            transitions=layer.replay_memory.sample(),
            reward_generator=layer.action_compatibility,
        )
        return loss

    def train_all(self):
        for layer_idx, layer in enumerate(self.abstract_layers, start=1):
            for action in range(layer.action_space.n):
                self.train(layer_idx, action)

    def explore(
        self,
        layer_idx: int | None = None,
        randomness: float = 1.0,
        p_term: float = 0.1,
        episode_length: int = 1,
    ) -> tuple[ObsType, float, bool, bool, dict]:
        if layer_idx == 0:
            raise Warning("Exploring base layer works, but is likely a logical error")
        if layer_idx is None:
            layer_idx = len(self.layers) - 1

        env_state, info = self.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(episode_length):
            env_state, reward, term, trunc, info = self.execute_option(
                layer_idx=layer_idx,
                action=int(self.option_space[layer_idx].sample()),
                randomness=randomness,
                p_term=p_term,
            )
            accumulated_reward += float(reward)
            if term or trunc:
                break
        return env_state, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        return self.layers[-1].reset(seed=seed, options=options)

    def __repr__(self) -> str:
        params = {
            "0": str(self.environment),
            **{f"{i+1}": str(layer) for i, layer in enumerate(self.layers)},
        }
        return torch_style_repr(self.__class__.__name__, params)
