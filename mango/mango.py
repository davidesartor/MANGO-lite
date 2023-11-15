from __future__ import annotations
from dataclasses import dataclass, field
from itertools import chain
import random
from typing import Generic, Iterator, NamedTuple, Optional, Sequence, TypeVar
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .abstractions.actions import AbstractActions
from .policies.dynamicpolicies import DQnetPolicyMapper, DynamicPolicy
from .utils import ReplayMemory, Transition, torch_style_repr

ObsType = TypeVar("ObsType", bound=npt.NDArray)


@dataclass(eq=False, slots=True, repr=False)
class MangoEnv(Generic[ObsType]):
    environment: gym.Env[ObsType, int]
    obs: ObsType = field(init=False)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.environment.action_space  # type: ignore

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, dict]:
        self.obs, reward, term, trunc, info = self.environment.step(action)
        info["mango:trajectory"] = [self.obs]
        return self.obs, float(reward), term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        self.obs, info = self.environment.reset(seed=seed, options=options)
        return self.obs, info


@dataclass(eq=False, slots=True, repr=False)
class MangoLayer(Generic[ObsType]):
    abstract_actions: AbstractActions[ObsType]
    lower_layer: MangoLayer[ObsType] | MangoEnv[ObsType]
    randomness: float = 0.0

    obs: ObsType = field(init=False)
    policy: DynamicPolicy = field(init=False)
    intrinsic_reward_log: tuple[list[float], ...] = field(init=False)
    train_loss_log: tuple[list[float], ...] = field(init=False)
    replay_memory: ReplayMemory[Transition] = field(init=False)

    def __post_init__(self) -> None:
        self.policy = DQnetPolicyMapper(
            comand_space=self.action_space,
            action_space=self.lower_layer.action_space,
        )
        self.intrinsic_reward_log = tuple([] for _ in range(self.action_space.n))
        self.train_loss_log = tuple([] for _ in range(self.action_space.n))
        self.replay_memory = ReplayMemory()

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.abstract_actions.action_space

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, dict]:
        trajectory = [self.obs]
        accumulated_reward = 0.0
        transitions = self.iterate_policy(comand=action)

        trajectory += list(
            chain.from_iterable(trans.info["mango:trajectory"] for trans in transitions)
        )
        accumulated_reward = sum(trans.reward for trans in transitions)
        term = any(trans.terminated for trans in transitions)
        trunc = any(trans.truncated for trans in transitions)
        info = {k: v for trans in transitions for k, v in trans.info.items()}
        info["mango:trajectory"] = trajectory
        self.replay_memory.extend(transitions)

        self.intrinsic_reward_log[action].append(
            self.abstract_actions.compatibility(action, trajectory[0], trajectory[-1])
        )
        return self.obs, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        self.obs, info = self.lower_layer.reset(seed=seed, options=options)
        return self.obs, info

    def train(self, action: Optional[int] = None):
        actions_to_train = range(self.action_space.n) if action is None else [action]
        for action in actions_to_train:
            loss = self.policy.train(
                comand=action,
                transitions=self.replay_memory.sample(),
                reward_generator=self.abstract_actions.compatibility,
            )
            self.train_loss_log[action].append(loss)

    def iterate_policy(self, comand: int) -> Iterator[Transition]:
        start_obs = self.obs
        while True:
            low_action = self.policy.get_action(
                comand=comand, state=start_obs, randomness=self.randomness
            )
            next_obs, reward, term, trunc, info = self.lower_layer.step(low_action)
            yield Transition(start_obs, low_action, next_obs, reward, term, trunc, info)
            if term or trunc:
                break
            if random.random() < self.abstract_actions.beta(self.obs, next_obs):
                break
            self.obs = start_obs = next_obs
        self.obs = next_obs


class Mango(Generic[ObsType]):
    def __init__(
        self,
        environment: gym.Env[ObsType, int],
        abstract_actions: Sequence[AbstractActions[ObsType]],
    ) -> None:
        self.environment = MangoEnv(environment)
        self.abstract_layers: list[MangoLayer[ObsType]] = []
        for concept in abstract_actions:
            self.abstract_layers.append(MangoLayer(concept, self.layers[-1]))
        self.reset()

    @property
    def layers(self) -> tuple[MangoEnv[ObsType] | MangoLayer[ObsType], ...]:
        return (self.environment, *self.abstract_layers)

    @property
    def option_space(self) -> tuple[gym.spaces.Discrete, ...]:
        return tuple(layer.action_space for layer in self.layers)

    def set_randomness(self, randomness: float, layer: Optional[int] = None):
        if layer is 0:
            raise ValueError("Cannot set randomness of environment actions")
        layers_to_set = range(1, len(self.layers)) if layer is None else [layer]
        for layer in layers_to_set:
            self.abstract_layers[layer - 1].randomness = randomness

    def execute_option(
        self, layer: int, action: int
    ) -> tuple[ObsType, float, bool, bool, dict]:
        return self.layers[layer].step(action)

    def train(self, layer: Optional[int] = None, action: Optional[int] = None):
        if layer == 0:
            raise ValueError("Cannot train environment actions")
        if layer is None and action is not None:
            raise Warning("Training same action in all layers")
        layers_to_train = range(1, len(self.layers)) if layer is None else [layer]
        for layer in layers_to_train:
            self.abstract_layers[layer - 1].train(action)

    def explore(
        self, layer: Optional[int] = None, episode_length: int = 1
    ) -> tuple[ObsType, float, bool, bool, dict]:
        if layer == 0:
            raise Warning("Exploring base layer works, but is likely a logical error")
        if layer is None:
            layer = len(self.layers) - 1

        env_state, info = self.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(episode_length):
            env_state, reward, term, trunc, info = self.execute_option(
                layer=layer, action=int(self.option_space[layer].sample())
            )
            accumulated_reward += reward
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
