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
    batch_size: int = 128
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


def plot_trajectory(trajectory: list[int], grid_shape: tuple[int, int]):
    square = 512 / grid_shape[0]
    for start_obs, next_obs in zip(trajectory[:-1], trajectory[1:]):
        y1, x1 = np.unravel_index(start_obs, grid_shape)
        y2, x2 = np.unravel_index(next_obs, grid_shape)
        plt.plot(
            [x1 * square + square // 2, x2 * square + square // 2],
            [y1 * square + square // 2, y2 * square + square // 2],
            "k--",
        )


def obs2int(obs, env_shape, onehot=False):
    y, x = np.unravel_index(np.argmax(obs), obs.shape[1:]) if onehot else obs
    return int(y * env_shape[1] + x)


def get_qvals_debug(policy, obs_list: list[ObsType]) -> list[float]:
    policy.net.eval()
    obs_tensor = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=policy.device)
    qvals = policy.net(obs_tensor).max(axis=1)[0]
    return list(qvals.cpu().detach().numpy())

def get_all_coords(env_shape: tuple[int, int], one_hot=False) -> list[ObsType]:
    y_matrix, x_matrix = np.indices(env_shape)
    obs_list = []
    for y, x in zip(y_matrix.flatten(), x_matrix.flatten()):
        if not one_hot:
            obs_list.append(ObsType(np.array([y, x])))
        else:
            obs_list.append(np.zeros((1, *env_shape), dtype=np.uint8))
            obs_list[-1][0, y, x] = 1
    return obs_list


def plot_qval_heatmap(policy, env, mask):
    qvals = get_qvals_debug(policy, env.all_observations())
    qvals = np.array(qvals).reshape(env.unwrapped.nrow, env.unwrapped.ncol)
    plt.imshow(qvals, cmap="PiYG")
    # colorbar on bottom
    plt.colorbar(orientation="horizontal")
