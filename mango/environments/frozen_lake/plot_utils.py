import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np
from .wrappers import FrozenLakeWrapper
from . import Actions

from mango.mango import Mango
from mango.agents import Agent
from mango.protocols import ObsType
from mango.policies import DQNetPolicy


def plot_grid(env: FrozenLakeWrapper, cell_shape: tuple[int, int], alpha=0.2):
    grid_shape = env.unwrapped.desc.shape
    pixels_in_square = env.unwrapped.cell_size
    pixels_in_cell = tuple(s * c for s, c in zip(cell_shape, pixels_in_square))

    offset = tuple(int(s * 0.2) for s in pixels_in_square)
    width, height = tuple(int(c - 0.4 * s) for s, c in zip(pixels_in_square, pixels_in_cell))
    for x in range(grid_shape[0] // cell_shape[0]):
        for y in range(grid_shape[1] // cell_shape[1]):
            position = tuple(p * c + o for p, c, o in zip((x, y), pixels_in_cell, offset))
            plt.gca().add_patch(
                plt.Rectangle(position, width, height, fc="red", alpha=alpha)  # type: ignore
            )


def plot_trajectory(trajectory: list[ObsType] | list[int], env: FrozenLakeWrapper):
    # TODO: fix type hints, this only works for wrappers that implement the observation_inv property
    if trajectory:
        if not isinstance(trajectory[0], int):
            trajectory = [env.observation_inv(obs) for obs in trajectory]  # type: ignore
        square = env.unwrapped.cell_size
        for start_obs, next_obs in zip(trajectory[:-1], trajectory[1:]):
            y1, x1 = np.unravel_index(start_obs, env.unwrapped.desc.shape)  # type: ignore
            y2, x2 = np.unravel_index(next_obs, env.unwrapped.desc.shape)  # type: ignore
            plt.plot(
                [x1 * square[1] + square[1] // 2, x2 * square[1] + square[1] // 2],
                [y1 * square[0] + square[0] // 2, y2 * square[0] + square[0] // 2],
                "k--",
            )


def all_observations(env: FrozenLakeWrapper, mask=lambda x: x) -> tuple[list[ObsType], list[bool]]:
    # TODO: fix type hints, this only works for observation wrappers
    s = env.unwrapped.s
    obs_list = []
    valid_mask = []
    y_matrix, x_matrix = np.indices((env.unwrapped.nrow, env.unwrapped.ncol))
    for y, x in zip(y_matrix.flatten(), x_matrix.flatten()):
        env.unwrapped.s = int(y * env.unwrapped.ncol + x)
        obs_list.append(mask(env.observation(env.unwrapped.s)))  # type: ignore
        valid_mask.append(
            not (env.unwrapped.desc[y, x] == b"H" or env.unwrapped.desc[y, x] == b"G")
        )
    env.unwrapped.s = s
    return obs_list, valid_mask


def get_qval(policy: DQNetPolicy, obs_list: list[ObsType]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        best_qvals, actions = policy.qvalues(np.stack(obs_list), batched=True).max(dim=1)  # type: ignore
    return best_qvals.cpu().detach().numpy(), actions.cpu().detach().numpy()


def plot_qval_heatmap(
    policy: DQNetPolicy, all_obs_list: tuple[list[ObsType], list[bool]], env, vmin=-1, vmax=1
):
    obs_list, valid_mask = all_obs_list
    best_qvals, actions = get_qval(policy, obs_list)
    best_qvals[~np.array(valid_mask, dtype=bool)] = np.nan
    best_qvals = np.array(best_qvals).reshape(env.unwrapped.desc.shape)

    cmap = mpl.colormaps.get_cmap("RdYlGn")  # type: ignore
    cmap.set_bad(color="aqua")
    plt.imshow(best_qvals, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    Y, X = np.indices(env.unwrapped.desc.shape)
    for y, x, act, is_valid in zip(Y.flatten(), X.flatten(), actions, valid_mask):
        if is_valid:
            dy, dx = Actions.to_delta(act)
            # draw arrows for actions in the middle of the cell
            plt.annotate(
                "",
                xy=(x + 0.4 * dx, y + 0.4 * dy),
                xytext=(x - 0.4 * dx, y - 0.4 * dy),
                arrowprops=dict(width=1, headwidth=3, headlength=3),
            )
    return best_qvals


def plot_all_abstractions(mango: Mango, trajectory: list[ObsType] | list[int] = []):
    # TODO: fix type hints, this only works for specific Mango instances
    env: FrozenLakeWrapper = mango.environment.environment  # type: ignore
    plt.figure(figsize=(3 * len(mango.abstract_layers) + 3, 3))
    plt.subplot(1, len(mango.abstract_layers) + 1, 1)
    plt.title(f"Environment")
    plt.imshow(env.unwrapped.render())
    plot_trajectory(trajectory, env)
    plt.xticks([])
    plt.yticks([])
    for col, layer in enumerate(mango.abstract_layers, start=1):
        plt.subplot(1, len(mango.abstract_layers) + 1, col + 1)
        plt.title(f"Layer {col+1} Abstraction")
        plt.imshow(env.render())  # type: ignore
        plot_trajectory(trajectory, env)
        plot_grid(env, layer.abs_actions.cell_shape)  # type: ignore
        plt.xticks([])
        plt.yticks([])


def plot_all_qvals(mango: Mango, trajectory: list[ObsType] | list[int] = [], size=3):
    # TODO: fix type hints, this only works for specific Mango instances
    env: FrozenLakeWrapper = mango.environment.environment  # type: ignore
    plt.figure(figsize=((size + 1) * len(Actions) + size, size * len(mango.abstract_layers)))
    n_rows, n_cols = len(mango.abstract_layers), len(Actions) + 1
    for row, layer in enumerate(mango.abstract_layers):
        plt.subplot(n_rows, n_cols, row * (len(Actions) + 1) + 1)
        plt.title(f"Layer {row+1} Abstraction")
        plt.imshow(env.unwrapped.render())
        plot_trajectory(trajectory, env)
        plot_grid(env, layer.abs_actions.cell_shape)  # type: ignore
        plt.xticks([])
        plt.yticks([])
        for col, action in enumerate(Actions, start=2):
            plt.subplot(n_rows, n_cols, row * (len(Actions) + 1) + col)
            plt.title(f"Qvals AbsAction {action.name}")
            policy = layer.policy.policies[action]  # type: ignore
            all_obs_list = all_observations(env, lambda obs: layer.abs_actions.mask(action, obs))  # type: ignore
            plot_qval_heatmap(policy, all_obs_list, env)
            plt.xticks([])
            plt.yticks([])


def plot_all_qvals_agent(agent: Agent, trajectory: list[ObsType] | list[int] = [], size=3):
    # TODO: fix type hints, this only works for specific Mango instances
    env: FrozenLakeWrapper = agent.environment.environment  # type: ignore
    plt.figure(figsize=((size + 1) + size, size))

    plt.subplot(1, 2, 1)
    plt.title(f"Environment")
    plt.imshow(env.unwrapped.render())
    plot_trajectory(trajectory, env)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.title(f"Qvals")
    all_obs_list = all_observations(env, lambda obs: layer.abs_actions.mask(action, obs))  # type: ignore
    plot_qval_heatmap(agent.policy, all_obs_list, env)  # type: ignore
    plt.xticks([])
    plt.yticks([])
