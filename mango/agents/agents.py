import pickle
from typing import Any, Optional, Sequence
from .. import spaces
from ..mango import MangoEnv, Mango
from ..policies.policies import Policy, DQnetPolicy
from ..policies.experiencereplay import ReplayMemory, Transition
from ..utils import ObsType, ActType, torch_style_repr


class Agent:
    def __init__(
        self,
        mango: MangoEnv | Mango,
        policy_params: dict[str, Any] = dict(),
    ):
        self.mango = mango
        self.policy = DQnetPolicy(action_space=mango.action_space, **policy_params)
        self.replay_memory = ReplayMemory()
        self.train_loss_log = []
        self.reward_log = []

    def step(self, randomness=0.0) -> tuple[ObsType, float, bool, bool, dict]:
        start_obs = self.mango.obs
        action = self.policy.get_action(start_obs, randomness)
        next_obs, reward, term, trunc, info = self.mango.step(action)
        self.replay_memory.push(Transition(start_obs, action, next_obs, reward, term, trunc, info))
        self.reward_log.append(reward)
        return self.mango.obs, reward, term, trunc, info

    def train(self) -> float | None:
        if not self.replay_memory.can_sample():
            return None
        loss = self.policy.train(transitions=self.replay_memory.sample())
        if loss is not None:
            self.train_loss_log.append(loss)
        return loss

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

    def save_to(self, path: str, include_mango: bool = True):
        self.mango.reset()
        mango = self.mango
        if not include_mango:
            self.mango: Mango | MangoEnv = None  # type: ignore
            raise Warning("Mango not saved, this may cause problems when loading")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if not include_mango:
            self.mango = mango

    @classmethod
    def load_from(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
