from dataclasses import dataclass, field
from typing import Optional, Sequence
import random
import numpy as np
import torch

from mango.actions.abstract_actions import AbstractActions
from ..utils import TensorTransitionLists, Transition, ObsType, ActType


@dataclass(eq=False, slots=True, repr=True)
class ExperienceReplay:
    abs_actions: AbstractActions
    batch_size: int = 256
    capacity: int = 2**10
    memories: dict[ActType, list[Transition]] = field(init=False)

    def __post_init__(self):
        self.memories = {comand: [] for comand in self.abs_actions.action_space}

    def size(self, comand: ActType) -> int:
        return len(self.memories[comand])

    def can_sample(self, comand: ActType, quantity: Optional[int] = None) -> bool:
        return self.size(comand) >= (quantity or self.batch_size)

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
            *random.choices(self.memories[comand], k=(quantity or self.batch_size))
        )
        start_obs = torch.as_tensor(np.stack(start_obs), dtype=torch.get_default_dtype())
        action = torch.as_tensor(np.array(action), dtype=torch.int64)
        next_obs = torch.as_tensor(np.stack(next_obs), dtype=torch.get_default_dtype())
        reward = torch.as_tensor(np.array(reward), dtype=torch.float32)
        terminated = torch.as_tensor(np.array(terminated), dtype=torch.bool)
        truncated = torch.as_tensor(np.array(truncated), dtype=torch.bool)
        info = list(info)
        return TensorTransitionLists(
            start_obs, action, next_obs, reward, terminated, truncated, info
        )
