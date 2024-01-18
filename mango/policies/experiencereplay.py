from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import numpy.typing as npt
import torch
from mango.protocols import AbstractActions, ActType, Transition


def to_tensor(self, transitions: Sequence[Transition]):
    obs_dtype = torch.get_default_dtype()
    action_dtype = torch.int64
    device = next(self.net.parameters()).device
    start_obs, action, next_obs, reward, terminated, truncated = zip(*transitions)
    start_obs = torch.as_tensor(torch.stack(start_obs), dtype=obs_dtype, device=device)
    action = torch.as_tensor(np.array(action), dtype=action_dtype, device=device)
    next_obs = torch.as_tensor(np.stack(next_obs), dtype=obs_dtype, device=device)
    reward = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device)
    terminated = torch.as_tensor(np.array(terminated), dtype=torch.bool, device=device)
    return start_obs, action, next_obs, reward, terminated, truncated


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
        beta = self.abstract_actions.beta(self.comand, transition)
        return transition._replace(
            start_obs=self.abstract_actions.mask(self.comand, transition.start_obs),
            next_obs=self.abstract_actions.mask(self.comand, transition.next_obs),
            reward=self.abstract_actions.reward(self.comand, transition),
            terminated=transition.terminated or beta,
        )


@dataclass(eq=False, slots=True, repr=True)
class ExperienceReplay:
    batch_size: int = 64
    capacity: int = 1024 * 16
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
        return self.memory.size >= (quantity or self.batch_size)

    def update_priorities_last_sampled(self, temporal_difference: npt.NDArray[np.floating]) -> None:
        self.priorities[self.last_sampled] = np.abs(temporal_difference) ** self.alpha + 1e-8

    def push(self, transition: Transition) -> None:
        if self.transform is not None:
            transition = self.transform(transition)
        idx = self.memory.push(transition)
        self.priorities[idx] = np.max(self.priorities) or 1.0

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.push(transition)

    def sample(self, quantity: Optional[int] = None) -> list[Transition]:
        if not self.can_sample(quantity):
            raise ValueError("Not enough samples to sample from")

        sample_prob = self.priorities / np.sum(self.priorities)
        self.last_sampled = np.random.choice(
            len(sample_prob), size=(quantity or self.batch_size), p=sample_prob
        )
        return self.memory[self.last_sampled]
