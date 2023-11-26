from typing import NewType, TypeVar
import numpy.typing as npt
from matplotlib import pyplot as plt

import torch


# this is not a good way to type this version
# but it will minimize changes when addin support for generic types
ObsType = npt.NDArray | torch.Tensor
ActType = NewType("ActType", int)
OptionType = ActType | tuple[int, ActType]

T = TypeVar("T")


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


def smooth(signal, window=0.05):
    signal = [s for s in signal if s is not None]
    window = max(3, int(len(signal) * window))
    return [sum(signal[i : i + window]) / window for i in range(len(signal) - window)]


def plot_loss_reward(mango, actions, reward_lims=None, layers=None):
    plt.figure(figsize=(12, 6))
    for layer_idx in layers or range(1, len(mango.abstract_layers) + 1):
        layer = mango.abstract_layers[layer_idx - 1]
        for action in actions:
            plt.subplot(len(mango.abstract_layers), 3, 3 * (layer_idx - 1) + 1)
            plt.title(f"loss Layer {layer_idx}")
            plt.semilogy(smooth(layer.train_loss_log[action]), label=f"{action.name}")
            plt.legend()

            plt.subplot(len(mango.abstract_layers), 3, 3 * (layer_idx - 1) + 2)
            plt.title(f"reward Layer {layer_idx}")
            rewards = smooth(layer.intrinsic_reward_log[action])
            plt.plot(rewards, label=f"{action.name}")
            plt.plot(len(rewards) - 1, rewards[-1], "o", color=plt.gca().lines[-1].get_color())
            plt.legend()
            plt.ylim(reward_lims)

        plt.subplot(len(mango.abstract_layers), 3, 3 * (layer_idx - 1) + 3)
        plt.title(f"episode lenght Layer {layer_idx}")
        plt.plot(smooth(layer.episode_length_log))
