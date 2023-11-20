from typing import Any
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
import torch
from .. import spaces
from ..utils import ActType, ObsType
from ..actions.abstract_actions import Grid2dActions


def generate_map(size=8, p=0.8, mirror=False, random_start=False, hide_goal=False):
    desc = generate_random_map(size=size // 2 if mirror else size, p=p)
    if hide_goal:
        desc = list(map(lambda row: row.replace("G", "F"), desc))
    if random_start:
        desc = list(map(lambda row: row.replace("F", "S"), desc))
    if mirror:
        desc = [row[::-1] + row for row in desc[::-1] + desc]
    return desc


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        **kwargs,
    ):
        if map_name == "RANDOM":
            desc = generate_map(**kwargs)
        super().__init__(render_mode, desc, map_name, is_slippery)
        self.action_space = spaces.Discrete(4)


class ReInitOnReset(gym.Wrapper):
    def __init__(self, env: gym.Env, **init_kwargs):
        super().__init__(env)
        self.init_kwargs = init_kwargs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.env.__init__(**self.init_kwargs)
        return self.env.reset(seed=seed, options=options)


class CoordinateObservation(gym.ObservationWrapper):
    def __init__(self, env: FrozenLakeEnv, one_hot: bool = False):
        super().__init__(env)
        self.one_hot = one_hot
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)  # type: ignore
        self.observation_space = (
            gym.spaces.Box(low=0, high=max(map_shape), shape=(2,), dtype=np.uint8)
            if not one_hot
            else gym.spaces.Box(low=0, high=1, shape=(1, *map_shape), dtype=np.uint8)
        )

    def observation(self, observation: int) -> ObsType:
        y, x = divmod(self.unwrapped.s, self.unwrapped.ncol)  # type: ignore
        if not self.one_hot:
            return ObsType(np.array([y, x], dtype=np.uint8))
        one_hot = np.zeros((1, self.unwrapped.nrow, self.unwrapped.ncol), dtype=np.uint8)  # type: ignore
        one_hot[0, y, x] = 1
        return ObsType(one_hot)

    def observation_inv(self, obs: ObsType) -> int:
        y, x = np.unravel_index(np.argmax(obs[0]), obs.shape[1:]) if self.one_hot else obs
        return int(y * self.unwrapped.ncol + x)  # type: ignore


class TensorObservation(gym.ObservationWrapper):
    char2int = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}.get

    def __init__(self, env: gym.Env, one_hot: bool = False):
        super().__init__(env)
        self.one_hot = one_hot
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)  # type: ignore
        self.observation_space = (
            gym.spaces.Box(low=0, high=1, shape=(4, *map_shape), dtype=np.uint8)
            if one_hot
            else gym.spaces.Box(low=0, high=3, shape=(1, *map_shape), dtype=np.uint8)
        )

    def observation(self, observation: int) -> ObsType:
        map = [[self.char2int(el) for el in list(row)] for row in self.unwrapped.desc]  # type: ignore
        row, col = divmod(self.unwrapped.s, self.unwrapped.ncol)  # type: ignore
        map[row][col] = 0
        map = np.array(map, dtype=np.uint8)
        if not self.one_hot:
            return ObsType(map[None, :, :])
        one_hot_map = np.zeros((4, *map.shape), dtype=np.uint8)
        Y, X = np.indices(map.shape)
        one_hot_map[map, Y, X] = 1
        one_hot_map = one_hot_map[[0, 2, 3], :, :]  # remove the 1="F"|"S" channel
        return ObsType(one_hot_map)

    def observation_inv(self, obs: ObsType) -> int:
        agent_idx = np.argmax(obs[0]) if self.one_hot else np.argmin(obs[0])
        y, x = np.unravel_index(agent_idx, obs.shape[1:])
        return int(y * self.unwrapped.ncol + x)  # type: ignore


class RenderObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)  # type: ignore
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, *map_shape), dtype=np.uint8
        )

    def observation(self, observation: int) -> ObsType:
        render = self.unwrapped._render_gui(mode="rgb_array")  # type: ignore
        return ObsType(render)


def plot_grid(env, cell_shape: tuple[int, int], alpha=0.2):
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


def all_observations(env) -> tuple[list[ObsType], list[bool]]:
    s = env.unwrapped.s  # type: ignore
    obs_list = []
    valid_mask = []
    y_matrix, x_matrix = np.indices((env.unwrapped.nrow, env.unwrapped.ncol))  # type: ignore
    for y, x in zip(y_matrix.flatten(), x_matrix.flatten()):
        env.unwrapped.s = int(y * env.unwrapped.ncol + x)  # type: ignore
        obs_list.append(env.observation(env.unwrapped.s))
        valid_mask.append(not env.unwrapped.desc[y, x] == b"H")  # type: ignore
    env.unwrapped.s = s  # type: ignore
    return obs_list, valid_mask


def get_qval(policy, obs_list: list[ObsType]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        policy.net.eval()
        obs_tensor = torch.as_tensor(
            np.stack(obs_list), dtype=torch.float32, device=policy.device
        )
        best_qvals, actions = policy.net(obs_tensor).max(axis=1)
    return best_qvals.cpu().detach().numpy(), actions.cpu().detach().numpy()


def plot_qval_heatmap(policy, env, mask=lambda x: x, cmap="RdYlGn", **kwargs):
    grid_shape = env.unwrapped.desc.shape
    obs_list, valid_mask = all_observations(env)
    best_qvals, actions = get_qval(policy, [mask(obs) for obs in obs_list])
    best_qvals[~np.array(valid_mask, dtype=bool)] = np.nan
    best_qvals = np.array(best_qvals).reshape(grid_shape)

    cmap = mpl.colormaps.get_cmap(cmap)  # type: ignore
    cmap.set_bad(color="aqua")
    plt.imshow(best_qvals, cmap=cmap, **kwargs)
    plt.colorbar()
    Y, X = np.indices(grid_shape)
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
        for col, action in enumerate(Grid2dActions, start=2):
            plt.subplot(
                len(mango.abstract_layers),
                len(Grid2dActions) + 1,
                row * (len(Grid2dActions) + 1) + col,
            )
            plt.title(f"Qvals AbsAction {action.name}")
            policy = layer.policy.policies[ActType(action)]
            plot_qval_heatmap(policy, env, layer.abs_actions.mask, cmap="RdYlGn", vmin=-1, vmax=1)  # type: ignore