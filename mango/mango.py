from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from itertools import chain
import random
from typing import Any, Iterator, Optional, Sequence
import gymnasium as gym

from . import spaces
from .actions.abstract_actions import AbstractActions
from .policies.dynamicpolicies import DQnetPolicyMapper, DynamicPolicy
from .utils import ReplayMemory, Transition, ObsType, ActType, torch_style_repr


@dataclass(eq=False, slots=True, repr=False)
class MangoEnv:
    environment: gym.Env[ObsType, ActType]
    verbose_indent: Optional[int] = None
    obs: ObsType = field(init=False)

    @property
    def action_space(self) -> spaces.Discrete:
        return self.environment.action_space  # type: ignore

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.verbose_indent is not None:
            print("  " * self.verbose_indent + f"obs: {self.obs}, action {action}")
        obs, reward, term, trunc, info = self.environment.step(action)
        info["mango:trajectory"] = [self.obs, obs]
        self.obs = obs
        return self.obs, float(reward), term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        if self.verbose_indent is not None:
            print("  " * self.verbose_indent + f"resetting environment")
        self.obs, info = self.environment.reset(seed=seed, options=options)
        return self.obs, info

    def __repr__(self) -> str:
        return torch_style_repr(self.__class__.__name__, dict(environment=str(self.environment)))


@dataclass(eq=False, slots=True, repr=False)
class MangoLayer:
    abs_actions: AbstractActions
    lower_layer: MangoLayer | MangoEnv
    policy_params: InitVar[dict[str, Any]] = dict()
    verbose_indent: Optional[int] = None
    randomness: float = 0.0

    policy: DQnetPolicyMapper = field(init=False)
    obs: ObsType = field(init=False)
    replay_memory: ReplayMemory[Transition] = field(init=False)
    intrinsic_reward_log: tuple[list[float], ...] = field(init=False)
    train_loss_log: tuple[list[float], ...] = field(init=False)

    def __post_init__(self, policy_params):
        self.policy = DQnetPolicyMapper(
            comand_space=self.action_space,
            action_space=self.lower_layer.action_space,
            policy_params=policy_params,
            obs_transform=self.abs_actions.mask,
        )
        self.intrinsic_reward_log = tuple([] for _ in self.action_space)
        self.train_loss_log = tuple([] for _ in self.action_space)
        self.replay_memory = ReplayMemory()

    @property
    def action_space(self) -> spaces.Discrete:
        return self.abs_actions.action_space

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.verbose_indent is not None:
            print("  " * self.verbose_indent + f"obs: {self.obs}, action {action}")

        trajectory, reward, term, trunc, info = [self.obs], 0.0, False, False, {}
        for transition in self.iterate_policy(comand=action):
            self.replay_memory.push(transition)
            trajectory += transition.info["mango:trajectory"][1:]
            reward += transition.reward
            term = transition.terminated or term
            trunc = transition.truncated or trunc
            info.update(transition.info)
        info["mango:trajectory"] = trajectory

        if info["mango:terminated"]:
            self.intrinsic_reward_log[action].append(
                self.abs_actions.compatibility(action, trajectory[0], trajectory[-1])
            )
        return self.obs, reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        self.obs, info = self.lower_layer.reset(seed=seed, options=options)
        return self.obs, info

    def train(self, action: Optional[ActType] = None):
        to_train = self.action_space if action is None else [action]
        for action in to_train:
            loss = self.policy.train(
                comand=action,
                transitions=self.replay_memory.sample(),
                reward_generator=self.abs_actions.compatibility,
            )
            if loss is not None:
                self.train_loss_log[action].append(loss)

    def iterate_policy(self, comand: ActType) -> Iterator[Transition]:
        start_obs = self.obs
        while True:
            action = self.policy.get_action(comand, start_obs, self.randomness)
            next_obs, reward, term, trunc, info = self.lower_layer.step(action)
            mango_term, mango_trunc = self.abs_actions.beta(comand, start_obs, next_obs)
            info["mango:terminated"], info["mango:truncated"] = mango_term, mango_trunc
            yield Transition(start_obs, action, next_obs, reward, term, trunc, info)
            if term or trunc or mango_term or mango_trunc:
                break
            start_obs = next_obs
        self.obs = next_obs

    def __repr__(self) -> str:
        return torch_style_repr(
            self.__class__.__name__,
            dict(abs_actions=str(self.abs_actions), policy=str(self.policy)),
        )


class Mango:
    def __init__(
        self,
        environment: gym.Env[ObsType, ActType],
        abstract_actions: Sequence[AbstractActions],
        policy_params: dict[str, Any] = dict(),
        verbose=False,
    ) -> None:
        self.verbose = verbose
        indents = [2 * i if verbose else None for i in range(len(abstract_actions) + 1)]
        self.environment = MangoEnv(environment, verbose_indent=indents[-1])
        self.abstract_layers: list[MangoLayer] = []
        for actions, indent in zip(abstract_actions, reversed(indents[:-1])):
            self.abstract_layers.append(MangoLayer(actions, self.layers[-1], policy_params, indent))
        self.reset()

    @property
    def layers(self) -> tuple[MangoEnv | MangoLayer, ...]:
        return (self.environment, *self.abstract_layers)

    @property
    def option_space(self) -> tuple[spaces.Discrete, ...]:
        return tuple(layer.action_space for layer in self.layers)

    def set_randomness(self, randomness: float, layer: Optional[int] = None):
        if layer == 0:
            raise ValueError("Cannot set randomness of environment actions")
        layers_to_set = range(1, len(self.layers)) if layer is None else [layer]
        for layer in layers_to_set:
            self.abstract_layers[layer - 1].randomness = randomness

    def execute_option(
        self, layer: int, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict]:
        if self.verbose:
            print(f"MANGO: Executing option {action} at layer {layer}")
        obs, reward, term, trunc, info = self.layers[layer].step(action)
        if self.verbose:
            print(f"MANGO: Option results: {obs=}, {reward=}")
        return obs, reward, term, trunc, info

    def train(self, layer: Optional[int] = None, action: Optional[ActType] = None):
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

        obs, info = self.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(episode_length):
            obs, reward, term, trunc, info = self.execute_option(
                layer=layer, action=ActType(int(self.option_space[layer].sample()))
            )
            accumulated_reward += reward
            if term or trunc:
                break
        return obs, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        return self.layers[-1].reset(seed=seed, options=options)

    def __repr__(self) -> str:
        params = {f"{i+1}": str(layer) for i, layer in enumerate(self.layers)}
        return torch_style_repr(self.__class__.__name__, params)


