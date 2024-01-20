import torch
from mango.protocols import Transition, StackedTransitions


class Buffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.last_in = -1
        self.size = 0

    def push(self, idx: int, item: torch.Tensor):
        if not hasattr(self, "memory"):
            self.memory = torch.empty((self.capacity, *item.shape), dtype=item.dtype)
        self.size = min(self.size + 1, len(self.memory))
        self.memory[idx] = item

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.memory[idx]


class ExperienceReplay:
    def __init__(self, batch_size=64, capacity=1024 * 16, alpha=0.6):
        self.batch_size = batch_size
        self.capacity = capacity
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.memory = [Buffer(self.capacity) for _ in "sasrtt"]
        self.priorities = torch.zeros(self.capacity, dtype=torch.float32)

    def can_sample(self) -> bool:
        assert all(mem.size == self.memory[0].size for mem in self.memory)
        return self.memory[0].size >= self.batch_size

    def update_priorities_last_sampled(self, temporal_difference: torch.Tensor) -> None:
        prio = torch.abs(temporal_difference) ** self.alpha + 1e-8
        self.priorities[self.last_sampled] = prio

    def push(self, transition: Transition) -> None:
        idx = int(self.priorities.argmin())
        for mem, elem in zip(self.memory, transition):
            mem.push(idx, torch.as_tensor(elem))
        self.priorities[idx] = 1e8

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.push(transition)

    def sample(self, device=None) -> StackedTransitions:
        if not self.can_sample():
            raise ValueError("Not enough samples to sample from")
        self.last_sampled = torch.multinomial(self.priorities, self.batch_size, replacement=False)
        return StackedTransitions(
            *(mem[self.last_sampled].to(device, non_blocking=True) for mem in self.memory)
        )
