from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Optional, TypeVar
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
    batch_size: int = 64
    capacity: int = 2**13
    last: int = field(default=0, init=False)
    memory: list[T] = field(default_factory=list, init=False)

    @property
    def size(self) -> int:
        return len(self.memory)

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        return self.size >= (batch_size or self.batch_size)

    def push(self, item: T) -> None:
        if self.size < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.last] = item
            self.last = (self.last + 1) % self.capacity
            
    def extend(self, items: Iterator[T]) -> None:
        self.memory.extend(items)

    def sample(self, quantity: Optional[int] = None) -> list[T]:
        if not self.can_sample(quantity):
            return []
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

def smooth(signal, window=10):
    signal = [s for s in signal if s is not None]
    return [sum(signal[i : i + window]) / window for i in range(len(signal) - window)]


## ultis for frozen lake

from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def generate_map(size=8, p=0.8, mirror=False, random_start=False):
    env_description = generate_random_map(size=size // 2 if mirror else size, p=p)
    if random_start:
        env_description = list(map(lambda row: row.replace("F", "S"), env_description)) 
    if mirror:
        env_description = [row[::-1]+ row for row in env_description[::-1] + env_description]
    return env_description

def plot_grid(
    grid_shape: tuple[int, int],
    cell_shape: tuple[int, int],
):
    square = 512 / grid_shape[0]
    pixels_per_cell = tuple(square * s for s in cell_shape)

    offset = (int(square * 0.2), int(square * 0.2))
    width, height = tuple(
        int(cell_size - 0.4 * square) for cell_size in pixels_per_cell
    )

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
    square = 512 / grid_shape[0]
    for obs in trajectory:
        y1, x1 = np.unravel_index(start, grid_shape)
        y2, x2 = np.unravel_index(obs, grid_shape)
        plt.plot(
            [x1 * square + square // 2, x2 * square + square // 2],
            [y1 * square + square // 2, y2 * square + square // 2],
            "k--",
        )
        start = obs



