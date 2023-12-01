from dataclasses import InitVar, dataclass, field
from typing import Any, Optional, Sequence
from mango.policies.experiencereplay import ExperienceReplay
from mango.protocols import Environment, ObsType, Policy, Transition
from mango.utils.repr import torch_style_repr


@dataclass(eq=False, slots=True, repr=False)
class Agent:
    environment: Environment
    policy_cls: InitVar[type[Policy]]
    policy_params: InitVar[dict[str, Any]]

    policy: Policy = field(init=False)
    replay_memory: ExperienceReplay = field(init=False, repr=False)
    train_loss_log: list[float] = field(init=False, repr=False, default_factory=list)
    reward_log: list[float] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self, policy_cls: type[Policy], policy_params: dict[str, Any]):
        self.policy = policy_cls.make(action_space=self.environment.action_space, **policy_params)
        self.replay_memory = ExperienceReplay(alpha=0.6)

    def step(self, obs: ObsType, randomness=0.0) -> tuple[ObsType, float, bool, bool, dict]:
        action = self.policy.get_action(obs, randomness)
        next_obs, reward, term, trunc, info = self.environment.step(action)
        info = {k: v for k, v in info.items() if not k.startswith("mango")}
        self.replay_memory.push(Transition(obs, action, next_obs, reward, term, trunc, info))
        return next_obs, reward, term, trunc, info

    def train(self):
        if self.replay_memory.can_sample():
            train_info = self.policy.train(transitions=self.replay_memory.sample())
            self.train_loss_log.append(train_info.loss)

    def explore(
        self, episode_length: int, randomness: float = 0.0
    ) -> tuple[ObsType, float, bool, bool, dict]:
        obs, info = self.environment.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(episode_length):
            obs, reward, term, trunc, info = self.step(obs, randomness=randomness)
            accumulated_reward += reward
            if term or trunc:
                break
        self.reward_log.append(accumulated_reward)
        return obs, accumulated_reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        if options is not None and options.get("replay_memory", False):
            self.replay_memory.reset()
        return self.environment.reset(seed=seed, options=options)

    def __repr__(self) -> str:
        return torch_style_repr(self.__class__.__name__, dict(policy=str(self.policy)))
