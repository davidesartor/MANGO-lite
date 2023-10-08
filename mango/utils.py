from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar
from typing import Generic, NamedTuple
import numpy.typing as npt
import random


class Transition(NamedTuple):
    start_state: npt.NDArray
    action: int
    next_state: npt.NDArray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


T = TypeVar("T")


@dataclass(eq=False)
class ReplayMemory(Generic[T]):
    batch_size: int = 256
    capacity: int = 2**15
    last: int = field(default=0, init=False)
    memory: list[T] = field(default_factory=list, init=False)

    @property
    def size(self) -> int:
        return len(self.memory)

    def push(self, item: T) -> None:
        if self.size < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.last] = item
            self.last = (self.last + 1) % self.capacity

    def sample(self, quantity: Optional[int] = None) -> list[T]:
        return random.sample(self.memory, min(self.size, (quantity or self.batch_size)))


def add_indent(s: str, indent=2, skip_first=True) -> str:
    """Add indentation to all lines in a string."""
    s = "\n".join(" " * indent + line for line in s.splitlines())
    if skip_first:
        s = s[indent:]
    return s


def torch_style_repr(class_name: str, params: dict[str, str]) -> str:
    repr_str = class_name + "(\n"
    for k, v in params.items():
        repr_str += f"({k}): {v}\n"
    repr_str = add_indent(repr_str) + "\n)"
    return repr_str
