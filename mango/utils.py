from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, NamedTuple, NewType, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import random

import torch


# this is not a good way to type this version
# but it will minimize changes when addin support for generic types
ObsType = npt.NDArray | torch.Tensor
ActType = NewType("ActType", int)
OptionType = ActType | tuple[int, ActType]


T = TypeVar("T")


class Transition(NamedTuple):
    start_obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class TensorTransitionLists(NamedTuple):
    start_obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: list[dict[str, Any]]


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


def plot_loss_reward(mango, actions, reward_lims=None):
    plt.figure(figsize=(12, 6))
    for layer_idx, layer in enumerate(mango.abstract_layers, start=1):
        for action in actions:
            plt.subplot(len(mango.abstract_layers), 3, 3 * (layer_idx - 1) + 1)
            plt.title(f"loss Layer {layer_idx}")
            plt.semilogy(smooth(layer.train_loss_log[action]), label=f"{action.name}")
            plt.legend()

            plt.subplot(len(mango.abstract_layers), 3, 3 * (layer_idx - 1) + 2)
            plt.title(f"reward Layer {layer_idx}")
            plt.plot(smooth(layer.intrinsic_reward_log[action]), label=f"{action.name}")
            plt.legend()
            plt.ylim(reward_lims)

        plt.subplot(len(mango.abstract_layers), 3, 3 * (layer_idx - 1) + 3)
        plt.title(f"episode lenght Layer {layer_idx}")
        plt.plot(smooth(layer.episode_length_log))
