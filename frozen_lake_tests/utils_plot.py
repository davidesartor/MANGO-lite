import matplotlib.pyplot as plt
import numpy as np

from mango.actions import grid2d


def smooth(signal, window=0.05):
    if len(signal) < 10:
        return signal
    signal = np.array([s for s in signal if s is not None])
    window = max(3, int(len(signal) * window))
    window_array = np.ones(window) / window
    return np.convolve(signal, window_array, mode="valid")


def plot_mango_loss_reward(mango, gamma=0.75, layers=None, save_path=None):
    plt.figure(figsize=(12, 3 * (1 + len(mango.abstract_layers))))
    for layer_idx in layers or range(1, len(mango.abstract_layers) + 1):
        layer = mango.abstract_layers[layer_idx - 1]
        for action in grid2d.Actions:
            plt.subplot(len(mango.abstract_layers) + 1, 3, 3 * (layer_idx - 1) + 1)
            plt.title(f"loss Layer {layer_idx}")
            plt.semilogy(smooth(layer.train_loss_log[action]), label=f"{action.name}")
            plt.legend()
            plt.grid(True)

            plt.subplot(len(mango.abstract_layers) + 1, 3, 3 * (layer_idx - 1) + 2)
            plt.title(f"reward Layer {layer_idx}")
            rewards = smooth(layer.intrinsic_reward_log[action])
            plt.plot(rewards, label=f"{action.name}")
            plt.plot(len(rewards) - 1, rewards[-1], "o", color=plt.gca().lines[-1].get_color())  # type: ignore
            plt.legend()
            plt.grid(True)

        plt.subplot(len(mango.abstract_layers) + 1, 3, 3 * (layer_idx - 1) + 3)
        plt.title(f"episode lenght Layer {layer_idx}")
        plt.plot(smooth(layer.episode_length_log))
        plt.ylim((0, None))
        plt.grid(True)

    plt.subplot(len(mango.abstract_layers) + 1, 3, 3 * len(mango.abstract_layers) + 1)
    plt.title(f"loss Layer agent")
    plt.semilogy(smooth(mango.train_loss_log))
    plt.grid(True)

    plt.subplot(len(mango.abstract_layers) + 1, 3, 3 * len(mango.abstract_layers) + 2)
    plt.title(f"reward Layer agent")
    plt.plot(smooth(mango.reward_log[::2]), label="random")
    plt.plot(smooth(mango.reward_log[1::2]), label="evaluation")
    plt.legend()
    plt.grid(True)

    plt.subplot(len(mango.abstract_layers) + 1, 3, 3 * len(mango.abstract_layers) + 3)
    plt.title(f"episode lenght Layer agent")
    plt.plot(smooth(mango.episode_length_log[::2]), label="random")
    plt.plot(smooth(mango.episode_length_log[1::2]), label="evaluation")
    plt.ylim((0, None))
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_agent_loss_reward(agent, save_path=None):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.title(f"loss")
    plt.semilogy(smooth(agent.train_loss_log))
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.title(f"reward")
    plt.plot(smooth(agent.reward_log[::2]), label="random")
    plt.plot(smooth(agent.reward_log[1::2]), label="evaluation")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title(f"episode lenght")
    plt.plot(smooth(agent.episode_length_log[::2]), label="random")
    plt.plot(smooth(agent.episode_length_log[1::2]), label="evaluation")
    plt.ylim((0, None))
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
