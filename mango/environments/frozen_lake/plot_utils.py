import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np
from ...utils import ActType, ObsType
from ...actions.abstract_actions import Grid2dActions


def plot_grid(env, cell_shape: tuple[int, int], alpha=0.2):
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


def all_observations(env, mask=lambda x: x) -> tuple[list[ObsType], list[bool]]:
    s = env.unwrapped.s  # type: ignore
    obs_list = []
    valid_mask = []
    y_matrix, x_matrix = np.indices((env.unwrapped.nrow, env.unwrapped.ncol))  # type: ignore
    for y, x in zip(y_matrix.flatten(), x_matrix.flatten()):
        env.unwrapped.s = int(y * env.unwrapped.ncol + x)  # type: ignore
        obs_list.append(mask(env.observation(env.unwrapped.s)))
        valid_mask.append(not (env.unwrapped.desc[y, x] == b"H" or env.unwrapped.desc[y, x] == b"G"))  # type: ignore
    env.unwrapped.s = s  # type: ignore
    return obs_list, valid_mask


def get_qval(policy, obs_list: list[ObsType]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        policy.net.eval()
        obs_tensor = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=policy.device)
        best_qvals, actions = policy.net(obs_tensor).max(axis=1)
    return best_qvals.cpu().detach().numpy(), actions.cpu().detach().numpy()


def plot_qval_heatmap(policy, all_obs_list, env, cmap="RdYlGn", **kwargs):
    obs_list, valid_mask = all_obs_list
    best_qvals, actions = get_qval(policy, obs_list)
    best_qvals[~np.array(valid_mask, dtype=bool)] = np.nan
    best_qvals = np.array(best_qvals).reshape(env.unwrapped.desc.shape)

    cmap = mpl.colormaps.get_cmap(cmap)  # type: ignore
    cmap.set_bad(color="aqua")
    plt.imshow(best_qvals, cmap=cmap, **kwargs)
    plt.colorbar()
    Y, X = np.indices(env.unwrapped.desc.shape)
    for y, x, act, is_valid in zip(Y.flatten(), X.flatten(), actions, valid_mask):
        if is_valid:
            dy, dx = Grid2dActions.to_delta(act)
            # draw arrows for actions in the middle of the cell
            plt.annotate(
                "",
                xy=(x + 0.4 * dx, y + 0.4 * dy),
                xytext=(x - 0.4 * dx, y - 0.4 * dy),
                arrowprops=dict(width=1, headwidth=3, headlength=3),
            )
    return best_qvals


def plot_all_qvals(mango, env, trajectory=None, **kwargs):
    plt.figure(figsize=(4 * len(Grid2dActions) + 3, 3 * len(mango.abstract_layers)))
    for row, layer in enumerate(mango.abstract_layers):
        plt.subplot(
            len(mango.abstract_layers),
            len(Grid2dActions) + 1,
            row * (len(Grid2dActions) + 1) + 1,
        )
        plt.title(f"Layer {row+1} Abstraction")
        plt.imshow(env.render())  # type: ignore
        if trajectory is not None:
            plot_trajectory(trajectory, env)
        plot_grid(env, layer.abs_actions.cell_shape)  # type: ignore
        plt.xticks([])
        plt.yticks([])
        for col, action in enumerate(Grid2dActions, start=2):
            plt.subplot(
                len(mango.abstract_layers),
                len(Grid2dActions) + 1,
                row * (len(Grid2dActions) + 1) + col,
            )
            plt.title(f"Qvals AbsAction {action.name}")
            policy = layer.policy.policies[ActType(action)]
            plot_qval_heatmap(policy, all_observations(env, layer.abs_actions.mask), env, vmin=-1, vmax=1)  # type: ignore
            plt.xticks([])
            plt.yticks([])