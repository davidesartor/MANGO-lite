from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, NamedTuple
import math
from . import spaces, utils
from .protocols import AbstractAction, Policy
from .protocols import Environment, ObsType, ActType, Transition
from .policies.experiencereplay import ExperienceReplay, TransitionTransform


class MangoEnv:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    @property
    def action_space(self) -> spaces.Discrete:
        return self.environment.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self.environment.observation_space

    def step(self, comand: ActType, randomness=0.0) -> Transition:
        start_obs = self.obs
        self.obs, reward, term, trunc, info = self.environment.step(comand)
        return Transition(start_obs, comand, self.obs, reward, term, trunc)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        self.obs, info = self.environment.reset(seed=seed, options=options)
        return self.obs, info


class MangoLayer(MangoEnv):
    def __init__(
        self,
        lower_layer: MangoEnv,
        abstract_actions: list[AbstractAction],
        policies: list[Policy],
    ):
        if len(abstract_actions) != len(policies):
            raise ValueError("Mismatch between number of abstract actions and policies")
        self.environment = lower_layer
        self.abstract_actions = abstract_actions
        self.policies = policies

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.abstract_actions))

    @property
    def obs(self) -> ObsType:
        return self.environment.obs

    def step(self, comand: ActType, randomness=0.0) -> Transition:
        lower_steps: list[Transition] = []
        while True:
            obs_masked = self.abstract_actions[comand].mask(self.obs)
            action = self.policies[comand].get_action(obs_masked, randomness)
            lower_steps.append(self.environment.step(action))

            if lower_steps[-1].terminated or lower_steps[-1].truncated:
                break
            if self.abstract_actions[comand].beta(lower_steps[-1]):
                break
            # truncate episode when looping more than once
            if sum([(self.obs == step.start_obs).all() for step in lower_steps]) > 1:
                break

        return Transition.from_steps(comand, lower_steps)


class Mango:
    def __init__(
        self,
        environment: Environment,
        policy_cls: type[Policy],
        policy_params: dict[str, Any] = {},
    ) -> None:
        self.environment = MangoEnv(environment)
        self.abstract_layers: list[MangoLayer] = []
        self.policy = policy_cls.make(action_space=self.layers[-1].action_space, **policy_params)
        self.reset(options={"replay_memory": True, "logs": True})

    def add_abstract_layer(
        self,
        abstract_actions: AbstractActions,
        policy_cls: type[Policy],
        agent_policy_params: dict[str, Any] = {},
        layer_policy_params: dict[str, Any] = {},
    ) -> None:
        self.abstract_layers.append(
            MangoLayer(
                abs_actions=abstract_actions,
                lower_layer=self.layers[-1],
                dynamic_policy_cls=PolicyMapper,
                dynamic_policy_params=dict(
                    policy_cls=policy_cls, policy_params=layer_policy_params
                ),
            )
        )
        self.policy = policy_cls.make(
            action_space=self.layers[-1].action_space, **agent_policy_params
        )
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
