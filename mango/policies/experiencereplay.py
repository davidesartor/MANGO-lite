from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import numpy.typing as npt
import torch
from mango.protocols import AbstractActions, ActType, TensorTransitionLists, Transition


@dataclass(eq=False, slots=True, repr=True)
class CircularBuffer:
    capacity: int = 1024 * 16
    memory: npt.NDArray[Any] = field(init=False)
    last_in: int = field(init=False, default=-1)
    size: int = field(init=False, default=0)

    def __post_init__(self):
        self.memory = np.zeros(self.capacity, dtype=object)

    def push(self, item: Any) -> int:
        self.last_in = (self.last_in + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.memory[self.last_in] = item
        return self.last_in

    def __getitem__(self, index: int) -> Any:
        return self.memory[index]


@dataclass(eq=False, slots=True, repr=True)
class ExponentialBuffer:
    capacity: int = 1024 * 16
    memory: npt.NDArray[Any] = field(init=False)
    size: int = field(init=False, default=0)

    def __post_init__(self):
        self.memory = np.zeros(self.capacity, dtype=object)

    def push(self, item: Any) -> int:
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = np.random.randint(self.capacity)
        self.memory[idx] = item
        return idx

    def __getitem__(self, index: int) -> Any:
        return self.memory[index]


@dataclass(eq=False, slots=True, repr=True)
class TransitionTransform:
    abstract_actions: AbstractActions
    comand: ActType

    def __call__(self, transition: Transition) -> Transition:
        return transition._replace(
            start_obs=self.abstract_actions.mask(self.comand, transition.start_obs),
            next_obs=self.abstract_actions.mask(self.comand, transition.next_obs),
            reward=self.abstract_actions.compatibility(
                self.comand, transition.start_obs, transition.next_obs
            ),
        )


@dataclass(eq=False, slots=True, repr=True)
class ExperienceReplay:
    batch_size: int = 64
    capacity: int = 1024 * 16
    min_capacity: int = 64 * 64
    alpha: float = 0.6
    transform: Optional[Callable[[Transition], Transition]] = None
    memory: CircularBuffer = field(init=False)
    priorities: npt.NDArray[np.floating] = field(init=False)
    last_sampled: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.memory = CircularBuffer(capacity=self.capacity)
        self.priorities = np.zeros(self.capacity, dtype=np.floating)

    def can_sample(self, quantity: Optional[int] = None) -> bool:
        return self.memory.size >= (quantity or self.min_capacity)

    def update_priorities_last_sampled(self, temporal_difference: npt.NDArray[np.floating]) -> None:
        self.priorities[self.last_sampled] = np.abs(temporal_difference) ** self.alpha + 1e-8

    def push(self, transition: Transition) -> None:
        if self.transform is not None:
            transition = self.transform(transition)
        idx = self.memory.push(transition)
        self.priorities[idx] = np.max(self.priorities) or 1.0

    def sample(self, quantity: Optional[int] = None) -> TensorTransitionLists:
        if not self.can_sample(quantity):
            raise ValueError("Not enough samples to sample from")

        sample_prob = self.priorities / np.sum(self.priorities)
        self.last_sampled = np.random.choice(
            len(sample_prob), size=(quantity or self.batch_size), p=sample_prob
        )
        start_obs, action, next_obs, reward, terminated, truncated, info = zip(
            *self.memory[self.last_sampled]
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
