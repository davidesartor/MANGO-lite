from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Sequence

import torch

from mango.actions.abstract_actions import AbstractActions
from ..utils import TensorTransitionLists, Transition, ObsType, ActType


@dataclass(eq=False, slots=True, repr=True)
class ExperienceReplay:
    abs_actions: AbstractActions
    batch_size: int = 256
    capacity: int = 2**10
    memories: dict[ActType, list[Transition]] = field(default_factory=dict, init=False)

    @property
    def size(self) -> int:
        return min(len(memory) for comand, memory in self.memories.items())

    def can_sample(
        self, comand: Optional[ActType] = None, batch_size: Optional[int] = None
    ) -> bool:
        if comand is not None:
            return len(self.memories[comand]) >= (batch_size or self.batch_size)
        return self.size >= (batch_size or self.batch_size)

    def push(self, comand: ActType, transition: Transition) -> None:
        start_obs_masked = self.abs_actions.mask(transition.start_obs)
        next_obs_masked = self.abs_actions.mask(transition.next_obs)
        for comand, memory in self.memories.items():
            processed_transition = transition._replace(
                start_obs=start_obs_masked,
                next_obs=next_obs_masked,
                reward=self.abs_actions.compatibility(
                    comand, transition.start_obs, transition.next_obs
                ),
            )
            if len(memory) < self.capacity:
                memory.append(processed_transition)
            else:
                memory[np.random.randint(self.capacity)] = processed_transition

    def extend(self, comand: ActType, items: Sequence[Transition]) -> None:
        for item in items:
            self.push(comand, item)

    def sample(self, comand: ActType, quantity: Optional[int] = None) -> TensorTransitionLists:
        if not self.can_sample(comand, quantity):
            raise ValueError("Not enough samples to sample from")
        start_obs, action, next_obs, reward, terminated, truncated, info = zip(
            *np.random.choice(self.memories[comand], size=quantity or self.batch_size)
        )
        start_obs = torch.stack(start_obs)
        action = torch.tensor(action)
        next_obs = torch.stack(next_obs)
        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated)
        truncated = torch.tensor(truncated)
        info = {key: torch.tensor(value) for key, value in info[0].items()}
        return TensorTransitionLists(
            start_obs, action, next_obs, reward, terminated, truncated, info
        )
