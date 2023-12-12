from __future__ import annotations
from typing import Any, Optional, Sequence

from . import spaces, utils
from .protocols import Environment, AbstractActions, DynamicPolicy, Policy
from .protocols import ObsType, ActType, Transition
from .policies.experiencereplay import ExperienceReplay, TransitionTransform
from .policies.policymapper import PolicyMapper


class MangoEnv(Environment):
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.environment.action_space.n + 1)

    @property
    def observation_space(self) -> spaces.Space:
        return self.environment.observation_space

    def step(self, action: ActType, obs: ObsType) -> tuple[ObsType, float, bool, bool, dict]:
        if action in self.environment.action_space:
            next_obs, reward, term, trunc, info = self.environment.step(action)
        else:
            next_obs, reward, term, trunc, info = obs, 0.0, False, False, {}
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
            lower_action = self.policy.get_action(
                comand=action, obs=masked_obs, randomness=self.randomness
            )
            next_obs, reward, term, trunc, info = self.lower_layer.step(
                action=lower_action, obs=obs
            )
            transition = Transition(obs, lower_action, next_obs, reward, term, trunc, info)
            mango_term, mango_trunc = self.abs_actions.beta(action, transition)
            info["mango:terminated"], info["mango:truncated"] = mango_term, mango_trunc
            for replay_memory in self.replay_memory.values():
                replay_memory.push(transition)
            trajectory += info["mango:trajectory"][1:]
            obs = next_obs
            accumulated_reward += reward
            if term or trunc or mango_term or mango_trunc:
                break

        info.update({"mango:trajectory": trajectory})
        if mango_term or term or trunc:
            intrinsic_reward = self.abs_actions.reward(action, transition)
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
        policy_cls: type[Policy],
        dynamic_policy_cls: type[DynamicPolicy] = PolicyMapper,
        policy_params: dict[str, Any] = {},
        dynamic_policy_params: dict[str, Any] | Sequence[dict[str, Any]] = {},
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
        self.policy = policy_cls.make(action_space=self.layers[-1].action_space, **policy_params)
        self.reset(options={"replay_memory": True, "logs": True})

    @property
    def layers(self) -> tuple[MangoEnv | MangoLayer, ...]:
        return (self.environment, *self.abstract_layers)

    def step(self, obs: ObsType, randomness=0.0) -> tuple[ObsType, float, bool, bool, dict]:
        action = self.policy.get_action(obs, randomness)
        next_obs, reward, term, trunc, info = self.layers[-1].step(action, obs)
        info["mango:terminated"] = info["mango:terminated"] = False
        self.replay_memory.push(Transition(obs, action, next_obs, reward, term, trunc, info))
        return next_obs, reward, term, trunc, info

    def run_episode(
        self,
        randomness: float = 0.0,
        episode_length: Optional[int] = None,
    ) -> tuple[float, dict]:
        obs, info = self.environment.reset()
        accumulated_reward = 0.0
        trajectory = [obs]
        while len(trajectory) < episode_length or episode_length is None:
            obs, reward, term, trunc, info = self.step(obs, randomness=randomness)
            accumulated_reward += reward
            trajectory += info["mango:trajectory"][1:]
            if term or trunc:
                break
        info.update({"mango:trajectory": trajectory})
        self.reward_log.append(accumulated_reward)
        self.episode_length_log.append(len(info["mango:trajectory"]))
        return accumulated_reward, info

    def train(self):
        if self.replay_memory.can_sample():
            train_info = self.policy.train(transitions=self.replay_memory.sample())
            self.replay_memory.update_priorities_last_sampled(train_info.td)
            self.train_loss_log.append(train_info.loss)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        if options is not None:
            if options.get("replay_memory", False):
                self.replay_memory = ExperienceReplay()
            if options.get("logs", False):
                self.reward_log = []
                self.train_loss_log = []
                self.episode_length_log = []
        return self.layers[-1].reset(seed=seed, options=options)

    def __repr__(self) -> str:
        params = {
            "policy": str(self.policy),
            **{f"{i+1}": str(layer) for i, layer in enumerate(self.layers)},
        }
        return utils.torch_style_repr(self.__class__.__name__, params)
