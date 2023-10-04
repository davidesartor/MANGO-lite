import random
from dataclasses import dataclass, field
from typing import Generic, Iterable, Optional, Protocol, TypeVar

T = TypeVar("T")


class ReplayMemory(Protocol[T]):
    def push(self, item: T) -> None:
        ...

    def sample(self, quantity: Optional[int] = None) -> Iterable[T]:
        ...


@dataclass(eq=False)
class ListReplayMemory(Generic[T]):
    capacity: int = 2**15
    batch_size: int = 256
    memory: list[T] = field(default_factory=list, init=False)
    index: int = field(default=0, init=False)

    @property
    def size(self) -> int:
        return len(self.memory)

    def push(self, item: T) -> None:
        if self.size < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.index] = item
            self.index = (self.index + 1) % self.capacity

    def sample(self, quantity: Optional[int] = None) -> list[T]:
        return random.sample(self.memory, k=min(len(self.memory),(quantity or self.batch_size)))
