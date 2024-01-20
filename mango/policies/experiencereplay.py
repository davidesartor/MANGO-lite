import torch
from mango.protocols import Transition, StackedTransitions


class CircularBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.last_in = -1
        self.size = 0

    def push(self, item: torch.Tensor) -> int:
        if not hasattr(self, "memory"):
            self.memory = torch.empty((self.capacity, *item.shape), dtype=item.dtype)

        self.last_in = (self.last_in + 1) % len(self.memory)
        self.size = min(self.size + 1, len(self.memory))
        self.memory[self.last_in] = item
        return self.last_in

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.memory[idx]


class ExperienceReplay:
    def __init__(self, batch_size=64, capacity=1024 * 16, alpha=0.6, out_device=None):
        self.batch_size = batch_size
        self.capacity = capacity
        self.alpha = alpha
        self.out_device = out_device
        self.reset()

    def reset(self) -> None:
        self.memory = [CircularBuffer(self.capacity) for _ in "sasrtt"]
        self.priorities = torch.zeros(self.capacity, dtype=torch.float32)

    def can_sample(self) -> bool:
        assert all(mem.size == self.memory[0].size for mem in self.memory)
        return self.memory[0].size >= self.batch_size

    def update_priorities_last_sampled(self, temporal_difference: torch.Tensor) -> None:
        prio = torch.abs(temporal_difference) ** self.alpha + 1e-8
        self.priorities[self.last_sampled] = prio

    def push(self, transition: Transition) -> None:
        idxs = [mem.push(torch.as_tensor(elem)) for mem, elem in zip(self.memory, transition)]
        assert all(idx == idxs[0] for idx in idxs)
        self.priorities[idxs[0]] = self.priorities.max().clip(1.0)

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.push(transition)

    def sample(self) -> StackedTransitions:
        if not self.can_sample():
            raise ValueError("Not enough samples to sample from")
        self.last_sampled = torch.multinomial(self.priorities, self.batch_size, replacement=False)
        return StackedTransitions(
            *(mem[self.last_sampled].to(self.out_device, non_blocking=True) for mem in self.memory)
        )
