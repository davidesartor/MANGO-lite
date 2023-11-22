from typing import Any, Optional, Sequence
from .. import spaces
from ..mango import Mango
from ..policies.policies import Policy, DQnetPolicy
from ..utils import ReplayMemory, Transition, ObsType, ActType, torch_style_repr


class Agent:
    def __init__(self, mango: Mango, policy_params: dict[str, Any] = dict()):
        self.mango = mango
        self.policy = DQnetPolicy(action_space=mango.option_space, **policy_params)
        self.replay_memory = ReplayMemory()
        self.train_loss_log = []
        self.episode_reward_log = []

    def step(self, randomness=0.0) -> tuple[ObsType, float, bool, bool, dict]:
        start_obs = self.mango.obs
        option = self.policy.get_action(start_obs, randomness)
        next_obs, reward, term, trunc, info = self.mango.step(option)
        self.replay_memory.push(Transition(start_obs, option, next_obs, reward, term, trunc, info))
        return self.mango.obs, reward, term, trunc, info

    def train(self):
        loss = self.policy.train(transitions=self.replay_memory.sample())
        self.train_loss_log.append(loss)

    def explore(
        self, episode_length: int, randomness: float = 0.0
    ) -> tuple[ObsType, float, bool, bool, dict]:
        obs, info = self.mango.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(episode_length):
            obs, reward, term, trunc, info = self.step(randomness=randomness)
            accumulated_reward += reward
            if term or trunc:
                break
        return obs, accumulated_reward, term, trunc, info

    def __repr__(self) -> str:
        return torch_style_repr(self.__class__.__name__, dict(policy=str(self.policy)))
