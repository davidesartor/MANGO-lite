import matplotlib.pyplot as plt
import numpy as np


def smooth(signal, window=0.05):
    signal = np.array([s for s in signal if s is not None])
    window = max(3, int(len(signal) * window))
    window_array = np.ones(window) / window
    return np.convolve(signal, window_array, mode="valid")


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


def plot_agent_loss_reward(agent, reward_lims=None):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.title(f"loss")
    plt.semilogy(smooth(agent.train_loss_log))

    plt.subplot(1, 2, 2)
    plt.title(f"reward")
    plt.plot(smooth(agent.reward_log))
    plt.ylim(reward_lims)
