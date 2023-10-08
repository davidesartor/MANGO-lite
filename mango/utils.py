from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar
from typing import Generic, NamedTuple, SupportsFloat

import numpy as np
from matplotlib import pyplot as plt
import numpy.typing as npt
import random


class Transition(NamedTuple):
    start_state: npt.NDArray
    action: int
    next_state: npt.NDArray
    reward: SupportsFloat
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


def plot_grid(
    grid_shape: tuple[int, int],
    cell_shape: tuple[int, int],
):
    pixels_per_cell = tuple(64 * s for s in cell_shape)

    offset = (int(64 * 0.2), int(64 * 0.2))
    width, height = tuple(int(cell_size - 0.4 * 64) for cell_size in pixels_per_cell)

    for x in range(grid_shape[0] // cell_shape[0]):
        for y in range(grid_shape[1] // cell_shape[1]):
            position = (
                x * pixels_per_cell[0] + offset[0],
                y * pixels_per_cell[1] + offset[1],
            )
            plt.gca().add_patch(
                plt.Rectangle(position, width, height, fc="red", alpha=0.2)  # type: ignore
            )
            
def plot_trajectory(start: int, trajectory: list[int], grid_shape: tuple[int, int]):
    for obs in trajectory:
        y1, x1 = np.unravel_index(start, grid_shape)
        y2, x2 = np.unravel_index(obs, grid_shape)
        plt.plot([x1*64+32, x2*64+32], [y1*64+32, y2*64+32], "k--")
        start = obs
