from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from typing import Any, Optional, Sequence
import numpy as np

from . import spaces
from .utils.repr import torch_style_repr
from .protocols import Environment, AbstractActions, DynamicPolicy
from .protocols import ObsType, ActType, OptionType, Transition
from .policies.experiencereplay import ExperienceReplay, TransitionTransform
from .policies.policymapper import PolicyMapper


@dataclass(eq=False, slots=True, repr=False)
class MangoEnv(Environment):
    environment: Environment
    verbose_indent: Optional[int] = None
    obs: ObsType = field(init=False)

    @property
    def action_space(self) -> spaces.Discrete:
        return self.environment.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self.environment.observation_space

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.verbose_indent is not None:
            print("  " * self.verbose_indent + f"obs: {self.obs}, action {action}")
        next_obs, reward, term, trunc, info = self.environment.step(action)
        info["mango:trajectory"] = [self.obs, next_obs]
        info["mango:terminated"], info["mango:truncated"] = False, False
        self.obs = next_obs
        return next_obs, float(reward), term, trunc, info

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
class MangoLayer(Environment):
    lower_layer: MangoLayer | MangoEnv
    abs_actions: AbstractActions
    dynamic_policy_cls: InitVar[type[DynamicPolicy]]
    dynamic_policy_params: InitVar[dict[str, Any]]
    verbose_indent: Optional[int] = None
    randomness: float = 0.0

    policy: DynamicPolicy = field(init=False)
    replay_memory: dict[ActType, ExperienceReplay] = field(init=False)
    intrinsic_reward_log: tuple[list[float], ...] = field(init=False)
    train_loss_log: tuple[list[float], ...] = field(init=False)
    episode_length_log: list[int] = field(init=False)

    def __post_init__(
        self, dynamic_policy_cls: type[DynamicPolicy], dynamic_policy_params: dict[str, Any]
    ):
        self.policy = dynamic_policy_cls.make(
            comand_space=self.abs_actions.action_space,
            action_space=self.lower_layer.action_space,
            **dynamic_policy_params,
        )
        self.replay_memory = {
            act: ExperienceReplay(transform=TransitionTransform(self.abs_actions, act))
            for act in self.action_space
        }
        self.reset(options={"replay_memory": True, "logs": True})

    @property
    def action_space(self) -> spaces.Discrete:
        return self.abs_actions.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self.lower_layer.observation_space

    @property
    def obs(self) -> ObsType:
        return self.lower_layer.obs

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.verbose_indent is not None:
            print("  " * self.verbose_indent + f"obs: {self.obs}, action {action}")

        start_obs, trajectory, accumulated_reward = self.obs, [self.obs], 0.0
        while True:
            start_obs, masked_obs = self.obs, self.abs_actions.mask(action, self.obs)
            low_action = self.policy.get_action(action, masked_obs, self.randomness)
            next_obs, reward, term, trunc, info = self.lower_layer.step(action=low_action)
            mango_term, mango_trunc = self.abs_actions.beta(action, start_obs, next_obs)
            info["mango:terminated"], info["mango:truncated"] = mango_term, mango_trunc
            self.replay_memory[action].push(
                Transition(start_obs, low_action, next_obs, reward, term, trunc, info)
            )
            trajectory += info["mango:trajectory"][1:]
            accumulated_reward += reward
            if term or trunc or mango_term or mango_trunc:
                break

        info = {**info, "mango:trajectory": trajectory}
        if info["mango:terminated"] or term:  # or info["mango:truncated"] or trunc:
            self.intrinsic_reward_log[action].append(
                self.abs_actions.compatibility(action, trajectory[0], trajectory[-1])
            )
        return self.obs, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        if options is not None:
            if options.get("replay_memory", False):
                for memory in self.replay_memory.values():
                    memory.reset()
            if options.get("logs", False):
                self.intrinsic_reward_log = tuple([] for _ in self.action_space)
                self.train_loss_log = tuple([] for _ in self.action_space)
                self.episode_length_log = []
        return self.lower_layer.reset(seed=seed, options=options)

    def train(self, action: Optional[ActType | Sequence[ActType]] = None):
        if action is None:
            action = [act for act in self.action_space]
        to_train = action if isinstance(action, Sequence) else [action]
        for action in to_train:
            if self.replay_memory[action].can_sample():
                train_info = self.policy.train(
                    comand=action,
                    transitions=self.replay_memory[action].sample(),
                )
                self.train_loss_log[action].append(train_info.loss)
                self.replay_memory[action].update_priorities_last_sampled(train_info.td)

    def __repr__(self) -> str:
        return torch_style_repr(
            self.__class__.__name__,
            dict(abs_actions=str(self.abs_actions), policy=str(self.policy)),
        )


class Mango(Environment):
    def __init__(
        self,
        environment: Environment,
        abstract_actions: Sequence[AbstractActions],
        dynamic_policy_params: dict[str, Any] | Sequence[dict[str, Any]],
        dynamic_policy_cls: type[DynamicPolicy] = PolicyMapper,
        verbose=False,
    ) -> None:
        if not isinstance(dynamic_policy_params, Sequence):
            dynamic_policy_params = [dynamic_policy_params for _ in abstract_actions]
        indents = [2 * i if verbose else None for i in range(len(abstract_actions) + 1)]
        self.verbose = verbose
        self.environment = MangoEnv(environment, verbose_indent=indents[-1])
        self.abstract_layers: list[MangoLayer] = []
        for actions, indent, params in zip(
            abstract_actions, reversed(indents[:-1]), dynamic_policy_params
        ):
            self.abstract_layers.append(
                MangoLayer(
                    abs_actions=actions,
                    lower_layer=self.layers[-1],
                    dynamic_policy_cls=dynamic_policy_cls,
                    dynamic_policy_params=params,
                    verbose_indent=indent,
                )
            )
        self.reset()

    @property
    def obs(self) -> ObsType:
        return self.environment.obs

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(sum(int(layer.action_space.n) for layer in self.layers))

    @property
    def observation_space(self) -> spaces.Space:
        return self.environment.observation_space

    @property
    def layers(self) -> tuple[MangoEnv | MangoLayer, ...]:
        return (self.environment, *self.abstract_layers)

    def relative_option_idx(self, option: OptionType) -> tuple[int, ActType]:
        if isinstance(option, tuple):
            return option
        offsets = np.cumsum([layer.action_space.n for layer in self.layers])
        offsets = [0] + list(offsets)
        layer = int(np.searchsorted(offsets, option + 1)) - 1
        action = option - offsets[layer]
        return layer, action

    def set_randomness(self, randomness: float, layer: Optional[int] = None):
        if layer == 0:
            raise ValueError("Cannot set randomness of environment actions")
        layers_to_set = range(1, len(self.layers)) if layer is None else [layer]
        for layer in layers_to_set:
            self.abstract_layers[layer - 1].randomness = randomness

    def step(self, option: OptionType) -> tuple[ObsType, float, bool, bool, dict]:
        layer, action = self.relative_option_idx(option)
        if self.verbose:
            print(f"MANGO: Executing option {action} at layer {layer}")
        obs, reward, term, trunc, info = self.layers[layer].step(action)
        if self.verbose:
            print(f"MANGO: Option results: {obs=}, {reward=}")
        return obs, reward, term, trunc, info

    def train(
        self,
        layer: Optional[int | Sequence[int]] = None,
        options: Optional[Sequence[OptionType]] = None,
    ):
        if options is not None:
            if layer is not None:
                raise ValueError("Cannot specify both layer and options")
            to_train = [self.relative_option_idx(o) for o in options]
        else:
            if layer is None:
                layer = range(1, len(self.layers))
            elif isinstance(layer, int):
                layer = [layer]
            to_train = [(l, None) for l in layer]

        for layer, action in to_train:
            if layer == 0:
                raise ValueError("Cannot train environment actions")
            self.abstract_layers[layer % len(self.layers) - 1].train(action)

    def explore(
        self, layer: Optional[int] = None, episode_length: int = 1
    ) -> tuple[ObsType, float, bool, bool, dict]:
        if layer == 0:
            raise ValueError("Environment actions do not need to be explored")
        if layer is None:
            layer = len(self.layers) - 1

        obs, info = self.reset()
        accumulated_reward, term, trunc, i = 0.0, False, False, 0
        for i in range(episode_length):
            action = self.layers[layer].action_space.sample()
            obs, reward, term, trunc, info = self.step((layer, action))
            accumulated_reward += reward
            if term or trunc:
                break
        self.abstract_layers[layer % len(self.layers) - 1].episode_length_log.append(i + 1)
        return obs, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        return self.layers[-1].reset(seed=seed, options=options)

    def __repr__(self) -> str:
        params = {f"{i+1}": str(layer) for i, layer in enumerate(self.layers)}
        return torch_style_repr(self.__class__.__name__, params)
