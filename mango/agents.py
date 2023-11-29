from dataclasses import InitVar, dataclass, field
from typing import Any, Optional, Sequence
from mango.policies.experiencereplay import ReplayMemory
from mango.protocols import Environment, ObsType, Policy, Transition
from mango.utils.repr import torch_style_repr


@dataclass(eq=False, slots=True, repr=False)
class Agent:
    environment: Environment
    policy_cls: InitVar[type[Policy]]
    policy_params: InitVar[dict[str, Any]]

    policy: Policy = field(init=False)
    replay_memory: ReplayMemory = field(init=False, repr=False, default_factory=ReplayMemory)
    train_loss_log: list[float] = field(init=False, repr=False, default_factory=list)
    reward_log: list[float] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self, policy_cls: type[Policy], policy_params: dict[str, Any]):
        self.policy = policy_cls.make(action_space=self.environment.action_space, **policy_params)

    def step(self, randomness=0.0) -> tuple[ObsType, float, bool, bool, dict]:
        start_obs = self.environment.obs
        action = self.policy.get_action(start_obs, randomness)
        next_obs, reward, term, trunc, info = self.environment.step(action)
        self.replay_memory.push(Transition(start_obs, action, next_obs, reward, term, trunc, info))
        self.reward_log.append(reward)
        return self.environment.obs, reward, term, trunc, info

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
        obs, info = self.environment.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(episode_length):
            obs, reward, term, trunc, info = self.step(randomness=randomness)
            accumulated_reward += reward
            if term or trunc:
                break
        return obs, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        if options is not None and options.get("replay_memory", False):
            self.replay_memory.reset()
        return self.environment.reset(seed=seed, options=options)

    def __repr__(self) -> str:
        return torch_style_repr(self.__class__.__name__, dict(policy=str(self.policy)))
