from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, NamedTuple, NewType, Optional, TypeVar

import numpy as np
from matplotlib import pyplot as plt
import random
import torch

# this is not a good way to type this version
# but it will minimize changes when addin support for generic types
ObsType = NewType("ObsType", np.ndarray)
ActType = NewType("ActType", int)
T = TypeVar("T")


class Transition(NamedTuple):
    start_obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(eq=False)
class ReplayMemory(Generic[T]):
    batch_size: int = 256
    capacity: int = 2**10
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


def plot_grid(env, cell_shape: tuple[int, int]):
    grid_shape = env.unwrapped.desc.shape
    pixels_in_square = env.unwrapped.cell_size
    pixels_in_cell = tuple(s * c for s, c in zip(cell_shape, pixels_in_square))

    offset = tuple(int(s * 0.2) for s in pixels_in_square)
    width, height = tuple(
        int(c - 0.4 * s) for s, c in zip(pixels_in_square, pixels_in_cell)
    )
    for x in range(grid_shape[0] // cell_shape[0]):
        for y in range(grid_shape[1] // cell_shape[1]):
            position = tuple(p * c + o for p, c, o in zip((x, y), pixels_in_cell, offset))
            plt.gca().add_patch(
                plt.Rectangle(position, width, height, fc="red", alpha=0.2)  # type: ignore
            )


def plot_trajectory(trajectory: list[ObsType] | list[int], env):
    if not isinstance(trajectory[0], int):
        trajectory = [env.observation_inv(obs) for obs in trajectory]
    square = env.unwrapped.cell_size
    for start_obs, next_obs in zip(trajectory[:-1], trajectory[1:]):
        y1, x1 = np.unravel_index(start_obs, env.unwrapped.desc.shape)
        y2, x2 = np.unravel_index(next_obs, env.unwrapped.desc.shape)
        plt.plot(
            [x1 * square[1] + square[1] // 2, x2 * square[1] + square[1] // 2],
            [y1 * square[0] + square[0] // 2, y2 * square[0] + square[0] // 2],
            "k--",
        )


def get_qvals_debug(policy, obs_list: list[ObsType]) -> list[float]:
    policy.net.eval()
    obs_tensor = torch.as_tensor(
        np.stack(obs_list), dtype=torch.float32, device=policy.device
    )
    qvals = policy.net(obs_tensor).max(axis=1)[0]
    return list(qvals.cpu().detach().numpy())


def plot_qval_heatmap(policy, env, mask=lambda x: x, **kwargs):
    qvals = get_qvals_debug(policy, [mask(obs) for obs in env.all_observations])
    qvals = np.array(qvals).reshape(env.unwrapped.nrow, env.unwrapped.ncol)
    plt.imshow(qvals, **kwargs)
    plt.colorbar()
    return qvals
