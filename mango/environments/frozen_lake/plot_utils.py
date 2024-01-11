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


def plot_map(env: FrozenLakeWrapper):
    plt.figure(figsize=(3, 3))
    plt.title(f"Environment")
    plt.imshow(env.unwrapped.render())
    plt.xticks([])
    plt.yticks([])


def plot_grid(env: FrozenLakeWrapper, cell_shape: tuple[int, int], alpha=0.2):
    grid_shape = env.unwrapped.desc.shape
    if env.unwrapped.fail_on_out_of_bounds:
        grid_shape = tuple(s - 1 for s in grid_shape)
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
            if env.unwrapped.fail_on_out_of_bounds:
                y1, x1, y2, x2 = y1 - 1, x1 - 1, y2 - 1, x2 - 1
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


def plot_qval_heatmap(policy: DQNetPolicy, all_obs_list: tuple[list[ObsType], list[bool]], env):
    grid_shape = env.unwrapped.desc.shape
    obs_list, valid_mask = all_obs_list
    best_qvals, actions = get_qval(policy, obs_list)
    best_qvals[~np.array(valid_mask, dtype=bool)] = np.nan
    valid_mask = np.array(valid_mask).reshape(grid_shape)
    best_qvals = np.array(best_qvals).reshape(grid_shape)
    if env.unwrapped.fail_on_out_of_bounds:
        grid_shape = tuple(s - 2 for s in grid_shape)
        best_qvals = best_qvals[1:-1, 1:-1]
        valid_mask = valid_mask[1:-1, 1:-1]

    cmap = mpl.colormaps.get_cmap("RdYlGn")  # type: ignore
    cmap.set_bad(color="aqua")
    vmax = max((0, np.nanmax(best_qvals)))
    vmin = min((0, np.nanmin(best_qvals)))
    plt.imshow(best_qvals, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    Y, X = np.indices(grid_shape)
    for y, x, act, is_valid in zip(Y.flatten(), X.flatten(), actions, valid_mask.flatten()):
        if is_valid:
            dy, dx = Actions.to_delta(act)
            # draw arrows for actions in the middle of the cell
            plt.annotate(
                "",
                xy=(x + 0.4 * dx, y + 0.4 * dy),
                xytext=(x - 0.4 * dx, y - 0.4 * dy),
                arrowprops=dict(width=1, headwidth=3, headlength=3),
            )
            # draw circles id dx == dy == 0
            if dx == 0 and dy == 0:
                plt.gca().add_patch(
                    plt.Circle((x, y), radius=0.2, fc="black", alpha=0.99)  # type: ignore
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
    plt.show()


def plot_all_qvals_mango_agent(
    mango_agent: Mango, trajectory: list[ObsType] | list[int] = [], size=3, save_path=None
):
    # TODO: fix type hints, this only works for specific Mango instances
    env: FrozenLakeWrapper = mango_agent.environment.environment  # type: ignore
    n_rows, n_cols = len(mango_agent.abstract_layers) + 1, len(Actions) + 1
    plt.figure(figsize=((size) * n_cols, size * n_rows))
    plt.subplot(n_rows, n_cols, 1)
    plt.title(f"Environment")
    plt.imshow(env.unwrapped.render())
    plot_trajectory(trajectory, env)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(n_rows, n_cols, 2)
    plt.title(f"Agent Qvals")
    all_obs_list = all_observations(env)  # type: ignore
    plot_qval_heatmap(mango_agent.policy, all_obs_list, env)  # type: ignore
    plt.xticks([])
    plt.yticks([])
    for row, layer in enumerate(reversed(mango_agent.abstract_layers), start=1):
        plt.subplot(n_rows, n_cols, row * (len(Actions) + 1) + 1)
        plt.title(f"Layer {n_rows-row} Abstraction")
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

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_all_qvals_normal_agent(
    agent: Agent, trajectory: list[ObsType] | list[int] = [], size=3, save_path=None
):
    # TODO: fix type hints, this only works for specific Mango instances
    env: FrozenLakeWrapper = agent.environment  # type: ignore
    plt.figure(figsize=((size + 2) + size, size))

    plt.subplot(1, 2, 1)
    plt.title(f"Environment")
    plt.imshow(env.unwrapped.render())
    plot_trajectory(trajectory, env)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.title(f"Qvals")
    all_obs_list = all_observations(env)  # type: ignore
    plot_qval_heatmap(agent.policy, all_obs_list, env)  # type: ignore
    plt.xticks([])
    plt.yticks([])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
