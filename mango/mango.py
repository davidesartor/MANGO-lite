from __future__ import annotations
from typing import Any, Optional, Sequence
import numpy as np

from . import spaces, utils
from .protocols import Environment, AbstractActions, DynamicPolicy
from .protocols import ObsType, ActType, OptionType, Transition
from .policies.experiencereplay import ExperienceReplay, TransitionTransform
from .policies.policymapper import PolicyMapper


class MangoEnv(Environment):
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    @property
    def action_space(self) -> spaces.Discrete:
        return self.environment.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self.environment.observation_space

    def step(self, action: ActType, obs: ObsType) -> tuple[ObsType, float, bool, bool, dict]:
        next_obs, reward, term, trunc, info = self.environment.step(action)
        info["mango:trajectory"] = [obs, next_obs]
        info["mango:terminated"] = False
        info["mango:truncated"] = False
        return next_obs, float(reward), term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        return self.environment.reset(seed=seed, options=options)

    def __repr__(self) -> str:
        return utils.torch_style_repr(
            self.__class__.__name__, dict(environment=str(self.environment))
        )


class MangoLayer(MangoEnv):
    def __init__(
        self,
        lower_layer: MangoEnv,
        abs_actions: AbstractActions,
        dynamic_policy_cls: type[DynamicPolicy],
        dynamic_policy_params: dict[str, Any],
        randomness: float = 0.0,
    ):
        self.lower_layer = lower_layer
        self.abs_actions = abs_actions
        self.policy = dynamic_policy_cls.make(
            comand_space=self.abs_actions.action_space,
            action_space=self.lower_layer.action_space,
            **dynamic_policy_params,
        )
        self.set_randomness(randomness)
        self.reset(options={"replay_memory": True, "logs": True})

    @property
    def action_space(self) -> spaces.Discrete:
        return self.abs_actions.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self.lower_layer.observation_space

    def set_randomness(self, randomness: float):
        self.randomness = randomness

    def step(self, action: ActType, obs: ObsType) -> tuple[ObsType, float, bool, bool, dict]:
        trajectory = [obs]
        accumulated_reward = 0.0
        while True:
            masked_obs = self.abs_actions.mask(action, obs)
            low_action = self.policy.get_action(action, masked_obs, self.randomness)
            next_obs, reward, term, trunc, info = self.lower_layer.step(action=low_action, obs=obs)
            mango_term, mango_trunc = self.abs_actions.beta(action, obs, next_obs)
            info["mango:terminated"], info["mango:truncated"] = mango_term, mango_trunc
            for replay_memory in self.replay_memory.values():
                replay_memory.push(
                    Transition(obs, low_action, next_obs, reward, term, trunc, info),
                )
            trajectory += info["mango:trajectory"][1:]
            obs = next_obs
            accumulated_reward += reward
            if term or trunc or mango_term or mango_trunc:
                break

        info.update({"mango:trajectory": trajectory})
        if mango_term or term:
            intrinsic_reward = self.abs_actions.compatibility(action, trajectory[0], trajectory[-1])
            self.intrinsic_reward_log[action].append(intrinsic_reward)
            self.episode_length_log.append(len(trajectory))
        return obs, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        if options is not None:
            if options.get("replay_memory", False):
                self.replay_memory = {
                    act: ExperienceReplay(transform=TransitionTransform(self.abs_actions, act))
                    for act in self.action_space
                }
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
                self.replay_memory[action].update_priorities_last_sampled(train_info.td)
                self.train_loss_log[action].append(train_info.loss)

    def explore(self, episode_length: int = 1) -> tuple[float, dict]:
        obs, info = self.reset()
        accumulated_reward = 0.0
        trajectory = [obs]
        while len(trajectory) < episode_length:
            action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(action, obs)
            accumulated_reward += reward
            trajectory += info["mango:trajectory"][1:]
            if term or trunc:
                break
        info.update({"mango:trajectory": trajectory})
        return accumulated_reward, info

    def __repr__(self) -> str:
        return utils.torch_style_repr(
            self.__class__.__name__,
            dict(abs_actions=str(self.abs_actions), policy=str(self.policy)),
        )


class Mango(MangoEnv):
    def __init__(
        self,
        environment: Environment,
        abstract_actions: Sequence[AbstractActions],
        dynamic_policy_params: dict[str, Any] | Sequence[dict[str, Any]],
        dynamic_policy_cls: type[DynamicPolicy] = PolicyMapper,
    ) -> None:
        if not isinstance(dynamic_policy_params, Sequence):
            dynamic_policy_params = [dynamic_policy_params for _ in abstract_actions]
        self.environment = MangoEnv(environment)
        self.abstract_layers: list[MangoLayer] = []
        for actions, params in zip(abstract_actions, dynamic_policy_params):
            self.abstract_layers.append(
                MangoLayer(
                    abs_actions=actions,
                    lower_layer=self.layers[-1],
                    dynamic_policy_cls=dynamic_policy_cls,
                    dynamic_policy_params=params,
                )
            )
        self.reset()

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

    def step(self, option: OptionType, obs: ObsType) -> tuple[ObsType, float, bool, bool, dict]:
        layer, action = self.relative_option_idx(option)
        return self.layers[layer].step(action=action, obs=obs)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        return self.layers[-1].reset(seed=seed, options=options)

    def set_randomness(self, randomness: float):
        for layer in self.abstract_layers:
            layer.set_randomness(randomness)

    def __repr__(self) -> str:
        params = {f"{i+1}": str(layer) for i, layer in enumerate(self.layers)}
        return utils.torch_style_repr(self.__class__.__name__, params)
