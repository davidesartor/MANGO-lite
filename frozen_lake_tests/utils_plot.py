from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from mango.actions import grid2D
from mango import Mango, Agent


def smooth(signal, window=0.05):
    window = max(3, int(len(signal) * window))
    if len(signal) < 10:
        return signal
    signal = np.array([s for s in signal if s is not None])
    window_array = np.ones(window) / window
    return np.convolve(signal, window_array, mode="valid")


def plot_mango_agent_loss_reward(
    mango: Mango, plot_inner_layers=True, save_path: Optional[str] = None
):
    nrows = len(mango.abstract_layers) + 1 if plot_inner_layers else 1
    ncols = 3
    plt.figure(figsize=(4 * ncols, 3 * nrows))

    if plot_inner_layers:
        for layer_idx, layer in enumerate(mango.abstract_layers, start=1):
            for action in grid2D.Actions:
                plt.subplot(nrows, ncols, 3 * (layer_idx - 1) + 1)
                plt.title(f"loss Layer {layer_idx}")
                plt.semilogy(smooth(layer.train_loss_log[action]), label=f"{action.name}")
                plt.legend()
                plt.grid(True)

                plt.subplot(nrows, ncols, 3 * (layer_idx - 1) + 2)
                plt.title(f"reward Layer {layer_idx}")
                rewards = smooth(layer.intrinsic_reward_log[action])
                plt.plot(rewards, label=f"{action.name}")
                plt.legend()
                plt.grid(True)

            plt.subplot(nrows, ncols, 3 * (layer_idx - 1) + 3)
            plt.title(f"episode lenght Layer {layer_idx}")
            plt.plot(smooth(layer.episode_length_log))
            plt.ylim((0, None))
            plt.grid(True)

    plt.subplot(nrows, ncols, 3 * nrows - 2)
    plt.title(f"loss Layer agent")
    plt.semilogy(smooth(mango.train_loss_log))
    plt.grid(True)

    plt.subplot(nrows, ncols, 3 * nrows - 1)
    plt.title(f"reward Layer agent")
    plt.plot(smooth(mango.reward_log[::2]), label="random")
    plt.plot(smooth(mango.reward_log[1::2]), label="evaluation")
    plt.ylim((0, 1.05))
    plt.legend()
    plt.grid(True)

    plt.subplot(nrows, ncols, 3 * nrows)
    plt.title(f"episode lenght Layer agent")
    plt.plot(smooth(mango.episode_length_log[::2]), label="random")
    plt.plot(smooth(mango.episode_length_log[1::2]), label="evaluation")
    plt.ylim((0, None))
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_normal_agent_loss_reward(agent: Agent, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.title(f"loss")
    plt.semilogy(smooth(agent.train_loss_log))
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.title(f"reward")
    plt.plot(smooth(agent.reward_log[::2]), label="random")
    plt.plot(smooth(agent.reward_log[1::2]), label="evaluation")
    plt.ylim((0, 1.05))
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


def plot_confront_loss_reward(
    agents: list[Mango | Agent], labels: list[str], save_path: Optional[str] = None
):
    plt.figure(figsize=(10, 4))
    if save_path is not None:
        plt.suptitle(save_path[-10:-4])
    plt.subplot(1, 2, 1)
    plt.title(f"reward")
    for agent, label in zip(agents, labels):
        plt.plot(smooth(agent.reward_log[1::2]), label=label + " evaluation")
        plt.plot(
            smooth(agent.reward_log[::2]),
            color=plt.gca().lines[-1].get_color(),
            alpha=0.3,
            label=label + " exploration",
        )
    plt.ylim((-0.05, 1.05))
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title(f"episode lenght")
    for agent, label in zip(agents, labels):
        plt.plot(smooth(agent.episode_length_log[1::2]), label=label + " evaluation")
        plt.plot(
            smooth(agent.episode_length_log[::2]),
            color=plt.gca().lines[-1].get_color(),
            alpha=0.3,
            label=label + " exploration",
        )
    plt.ylim((-0.1, None))
    plt.grid(True)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_confront_loss_reward_avg(
    agents: list[list[Mango | Agent]],
    labels: list[str],
    colors=["tab:blue", "tab:orange"],
    save_path: Optional[str] = None,
):
    def get_statistics(agents, eval=True):
        reward_logs = []
        ep_len_logs = []
        for agent in agents:
            reward_logs.append(smooth(agent.reward_log[1::2] if eval else agent.reward_log[::2]))
            ep_len_logs.append(
                smooth(agent.episode_length_log[1::2] if eval else agent.episode_length_log[::2])
            )
        max_len = max(len(r) for r in reward_logs)
        reward_logs = [np.pad(r, (0, max_len - len(r)), mode="edge") for r in reward_logs]
        reward_mean = np.mean(reward_logs, axis=0)
        reard_ci95 = 1.96 * np.std(reward_logs, axis=0) / np.sqrt(len(reward_logs))
        max_len = max(len(r) for r in ep_len_logs)
        ep_len_logs = [np.pad(r, (0, max_len - len(r)), mode="edge") for r in ep_len_logs]
        ep_len_mean = np.mean(ep_len_logs, axis=0)
        ep_len_ci95 = 1.96 * np.std(ep_len_logs, axis=0) / np.sqrt(len(ep_len_logs))
        return reward_mean, reard_ci95, ep_len_mean, ep_len_ci95

    plt.figure(figsize=(10, 4))
    if save_path is not None:
        plt.suptitle(f"average across {len(agents[0])} runs")
    for agent_list, label, color in zip(agents, labels, colors):
        for eval in [True, False]:
            r_mean, r_ci, ep_mean, ep_ci = get_statistics(agent_list, eval)
            plt.subplot(1, 2, 1)
            plt.title(f"reward")
            plt.plot(
                r_mean,
                "-" if eval else ":",
                color=color,
                alpha=1 if eval else 0.5,
                label=label + (" evaluation" if eval else " exploration"),
            )
            plt.fill_between(
                range(len(r_mean)),
                r_mean - r_ci,
                r_mean + r_ci,
                color=color,
                alpha=0.3 if eval else 0.1,
            )
            plt.ylim((-0.05, 1.05))
            plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.title(f"episode lenght")
            plt.plot(
                ep_mean,
                "-" if eval else ":",
                color=color,
                label=label + (" evaluation" if eval else " exploration"),
                alpha=1 if eval else 0.5,
            )
            plt.fill_between(
                range(len(ep_mean)),
                ep_mean - ep_ci,
                ep_mean + ep_ci,
                color=color,
                alpha=0.3 if eval else 0.1,
            )
    plt.ylim((-0.1, None))
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
