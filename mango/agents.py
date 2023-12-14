from __future__ import annotations
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
        episode_length: Optional[int] = None,
    ) -> list[Transition]:
        obs, info = self.environment.reset()
        transitions = []
        while episode_length is None or len(transitions) < episode_length:
            action = self.policy.get_action(obs, randomness)
            next_obs, reward, term, trunc, info = self.environment.step(action)
            transitions.append(Transition(obs, action, next_obs, reward, term, trunc, info))
            if term or trunc:
                break
            obs = next_obs
        self.reward_log.append(sum([t.reward for t in transitions]))
        self.episode_length_log.append(len(transitions))
        for transition in transitions:
            self.replay_memory.push(transition)
        return transitions

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
