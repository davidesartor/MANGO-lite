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
    img_shape: tuple[int, int] = (512, 512),
):
    pixels_in_grid_square = tuple(
        (pixels_in_img // squares_in_grid)
        for pixels_in_img, squares_in_grid in zip(img_shape, grid_shape)
    )
    pixels_per_cell = tuple(
        pixels_in_square * squares_in_cell
        for pixels_in_square, squares_in_cell in zip(pixels_in_grid_square, cell_shape)
    )

    offset = tuple(int(square_size * 0.2) for square_size in pixels_in_grid_square)
    width, height = tuple(
        int(cell_size - 0.4 * square_size)
        for cell_size, square_size in zip(pixels_per_cell, pixels_in_grid_square)
    )

    for x, y in np.ndindex(grid_shape):
        position = (
            x * pixels_per_cell[0] + offset[0],
            y * pixels_per_cell[1] + offset[1],
        )
        plt.gca().add_patch(
            plt.Rectangle(position, width, height, fc="red", alpha=0.2)  # type: ignore
        )
