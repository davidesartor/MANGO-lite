from dataclasses import dataclass, field
import re
from turtle import st
from typing import Generic, Optional, SupportsFloat, TypeVar
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .concepts import Concept
from .utils import ReplayMemory, Transition
from .policies import Policy, DQnetPolicy
from .mango import Mango

ObsType = TypeVar("ObsType")


@dataclass(eq=False, slots=True)
class Agent(Generic[ObsType]):
    mango: Mango[ObsType] = field(repr=False)
    base_concept: Concept[ObsType] = field(repr=False)
    policy: Policy = field(init=False)
    replay_memory: ReplayMemory[Transition] = field(default_factory=ReplayMemory)

    def __post_init__(self):
        n_options = sum(action_space.n for action_space in self.mango.option_space)
        self.policy = DQnetPolicy(action_space=gym.spaces.Discrete(n_options))

    def relative_option_idx(self, global_option_idx: int) -> tuple[int, int]:
        action = global_option_idx
        layer_idx = 0
        for action_space in self.mango.option_space:
            if action < action_space.n:
                break
            action -= int(action_space.n)
            layer_idx += 1
        return action, layer_idx

    def step(
        self, env_state: ObsType, randomness: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        start_state = self.base_concept.abstract(env_state)
        action = self.policy.get_action(start_state)
        relative_action, layer_idx = self.relative_option_idx(action)
        env_state, reward, term, trunc, info = self.mango.execute_option(
            action=relative_action, layer_idx=layer_idx, randomness=randomness
        )
        next_state = self.base_concept.abstract(env_state)
        self.replay_memory.push(
            Transition(start_state, action, next_state, reward, term, trunc, info)
        )
        return env_state, reward, term, trunc, info

    def train(self) -> None:
        if self.replay_memory.size > 1:
            self.policy.train(self.replay_memory.sample())

    def explore(
        self, randomness: float, max_steps: int = 1
    ) -> tuple[ObsType, float, bool, bool, dict]:
        env_state, info = self.mango.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(max_steps):
            env_state, reward, term, trunc, info = self.step(
                env_state, randomness=randomness
            )
            accumulated_reward += float(reward)
            if term or trunc:
                break
        return env_state, accumulated_reward, term, trunc, info
