from dataclasses import dataclass, field
from typing import Any, Optional

from .. import spaces
from ..mango import Mango
from ..policies.policies import Policy, DQnetPolicy
from ..utils import ReplayMemory, Transition, ObsType, ActType, torch_style_repr


class Agent:
    def __init__(self, mango: Mango, policy_params: dict[str, Any] = dict()):
        self.mango = mango
        self.policy = DQnetPolicy(
            action_space=spaces.Discrete(mango.option_space.n), **policy_params
        )
        self.replay_memory = ReplayMemory()

    def step(
        self, env_state: ObsType, randomness: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        start_state = self.base_concept.abstract(env_state)
        action = self.policy.get_action(start_state)
        layer_idx, relative_action = self.relative_option_idx(action)
        env_state, reward, term, trunc, info = self.mango.execute_option(
            layer_idx=layer_idx, action=relative_action, randomness=randomness
        )
        next_state = self.base_concept.abstract(env_state)
        self.replay_memory.push(
            Transition(start_state, action, next_state, reward, term, trunc, info)
        )
        return env_state, reward, term, trunc, info

    def train(self, action: Optional[ActType] = None):
        to_train = self.action_space if action is None else [action]
        for action in to_train:
            loss = self.policy.train(
                comand=action,
                transitions=self.replay_memory.sample(),
                reward_generator=self.abs_actions.compatibility,
            )
            if loss is not None:
                self.train_loss_log[action].append(loss)
    def explore(
        self, randomness: float, max_steps: int = 1
    ) -> tuple[ObsType, float, bool, bool, dict]:
        env_state, info = self.mango.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(max_steps):
            env_state, reward, term, trunc, info = self.step(env_state, randomness=randomness)
            accumulated_reward += float(reward)
            if term or trunc:
                break
        return env_state, accumulated_reward, term, trunc, info
