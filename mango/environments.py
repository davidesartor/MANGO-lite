from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym


@dataclass(eq=False, slots=True)
class DummyEnvironment(gym.Env[int, int]):
    state: int = 0
    observation_space: gym.spaces.Space = gym.spaces.Discrete(2)
    action_space: gym.spaces.Discrete = gym.spaces.Discrete(2)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        reward = float(self.state == action)
        self.state = self.observation_space.sample()
        return self.state, reward, False, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[int, dict]:
        self.state = self.observation_space.sample()
        return self.state, {}

    def render(self, mode: str = "human"):
        ...
