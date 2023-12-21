from __future__ import annotations
import math
from typing import Any, Optional
from mango.policies.experiencereplay import ExperienceReplay
from mango.protocols import Environment, ObsType, Policy, Transition
from mango.utils import torch_style_repr


class Agent:
    def __init__(
        self, environment: Environment, policy_cls: type[Policy], policy_params: dict[str, Any]
    ):
        self.environment = environment
        self.policy = policy_cls.make(action_space=self.environment.action_space, **policy_params)
        self.reset(options={"replay_memory": True, "logs": True})

    def run_episode(
        self,
        randomness: float = 0.0,
        episode_length=math.inf,
    ) -> tuple[list[ObsType], list[float]]:
        obs, info = self.environment.reset()
        trajectory = [obs]
        rewards = []
        while len(trajectory) < episode_length:
            action = self.policy.get_action(obs, randomness)
            next_obs, reward, term, trunc, info = self.environment.step(action)

            if randomness == 0.0:
                for seen_obs in trajectory:
                    if (seen_obs == next_obs).all():
                        trunc = True
            else:
                self.replay_memory.push(
                    Transition(obs, action, next_obs, reward, term, trunc, info)
                )
            trajectory.append(next_obs)
            rewards.append(reward)
            obs = next_obs

            if term or trunc:
                break

        self.reward_log.append(sum(rewards))
        self.episode_length_log.append(len(trajectory))
        return trajectory, rewards

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
        return self.environment.reset(seed=seed, options=options)

    def __repr__(self) -> str:
        return torch_style_repr(self.__class__.__name__, dict(policy=str(self.policy)))
