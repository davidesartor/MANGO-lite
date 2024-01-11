from __future__ import annotations
from typing import Any, Optional, Sequence, NamedTuple
import math
from . import spaces, utils
from .protocols import Environment, AbstractActions, DynamicPolicy, OptionTransition, Policy
from .protocols import ObsType, ActType, Transition
from .policies.experiencereplay import ExperienceReplay, TransitionTransform
from .policies.policymapper import PolicyMapper


class MangoEnv:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(int(self.environment.action_space.n) + 1)

    @property
    def observation_space(self) -> spaces.Space:
        return self.environment.observation_space

    def step(self, comand: ActType, randomness=0.0, episode_length=math.inf) -> OptionTransition:
        trajectory = [self.obs]
        if comand in self.environment.action_space:
            self.obs, reward, term, trunc, info = self.environment.step(comand)
            trajectory.append(self.obs)
        else:
            reward, term, trunc, info = 0.0, False, False, {}
        if episode_length <= 1:
            trunc = True
        return OptionTransition(
            comand=comand,
            trajectory=trajectory,
            rewards=[reward],
            failed=False,
            beta=True,
            terminated=term,
            truncated=trunc,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        self.obs, info = self.environment.reset(seed=seed, options=options)
        return self.obs, info

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
    ):
        self.lower_layer = lower_layer
        self.abs_actions = abs_actions
        self.policy = dynamic_policy_cls.make(
            comand_space=self.abs_actions.action_space,
            action_space=self.lower_layer.action_space,
            **dynamic_policy_params,
        )

    @property
    def action_space(self) -> spaces.Discrete:
        return self.abs_actions.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self.lower_layer.observation_space

    @property
    def obs(self) -> ObsType:
        return self.lower_layer.obs

    def step(self, comand: ActType, randomness=0.0, episode_length=math.inf) -> OptionTransition:
        trajectory = [self.obs]
        seen_obs = [self.obs]
        rewards = []
        while True:
            obs_masked = self.abs_actions.mask(comand, self.obs)
            action = self.policy.get_action(comand, obs_masked, randomness)
            lower_step = self.lower_layer.step(
                comand=action,
                randomness=randomness**2,
                episode_length=episode_length - len(trajectory),
            )
            trajectory += lower_step.trajectory[1:]
            rewards += lower_step.rewards

            term, trunc = lower_step.terminated, lower_step.truncated
            beta = self.abs_actions.beta(comand, lower_step.flatten())

            if sum([(self.obs == obs).all() for obs in seen_obs]) > 1:
                trunc = True
            seen_obs.append(self.obs)

            if randomness > 0.0:
                if lower_step.beta and not lower_step.failed:
                    for replay_memory in self.replay_memory.values():
                        replay_memory.extend(lower_step.all_transitions())

            if term or trunc or beta:
                break

        has_failed = self.abs_actions.has_failed(comand, trajectory[0], trajectory[-1])
        option_resuts = OptionTransition(comand, trajectory, rewards, has_failed, beta, term, trunc)

        if beta or term or trunc:
            intrinsic_reward = self.abs_actions.reward(comand, option_resuts.flatten())
            self.intrinsic_reward_log[comand].append(intrinsic_reward)
            self.episode_length_log.append(len(trajectory))

        return option_resuts

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

    @property
    def obs(self) -> ObsType:
        return self.environment.obs

    def run_episode(
        self, randomness: float = 0.0, episode_length=math.inf
    ) -> tuple[list[ObsType], list[float]]:
        self.environment.reset()
        trajectory = [self.obs]
        seen_obs = [self.obs]
        rewards = []
        while True:
            action = self.policy.get_action(self.obs, randomness)
            lower_step = self.layers[-1].step(
                comand=action,
                randomness=randomness**2,
                episode_length=episode_length - len(trajectory),
            )
            trajectory += lower_step.trajectory[1:]
            rewards += lower_step.rewards

            term, trunc = lower_step.terminated, lower_step.truncated

            if sum([(self.obs == obs).all() for obs in seen_obs]) > 1:
                trunc = True
            seen_obs.append(self.obs)

            if randomness > 0.0:
                if lower_step.beta and not lower_step.failed:
                    self.replay_memory.extend(lower_step.all_transitions())

            if term or trunc:
                break

        self.reward_log.append(sum(rewards))
        self.episode_length_log.append(len(trajectory))
        return trajectory, rewards

    def train(self, all_layers=True):
        if all_layers:
            for layer in self.abstract_layers:
                layer.train()
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
