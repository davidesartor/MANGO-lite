from typing import Any, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
import torch
from .. import spaces
from ..utils import ActType, ObsType
from ..actions.abstract_actions import Grid2dActions
import numba


@numba.njit
def sample_position_in(region: npt.NDArray[np.bool_]) -> tuple[int, int]:
    r, c = np.random.randint(region.shape[0]), np.random.randint(region.shape[1])
    if region.sum() == 0:
        return r, c
    while not region[r, c]:
        r, c = np.random.randint(region.shape[0]), np.random.randint(region.shape[1])
    return r, c


@numba.njit
def reachable_from(
    start: npt.NDArray[np.bool_], frozen: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    reached = start.copy() & frozen
    total = 0
    while total < reached.sum():
        total = reached.sum()
        adj = np.zeros(reached.shape, dtype=np.bool_)
        adj[1:, :] |= reached[:-1, :]
        adj[:-1, :] |= reached[1:, :]
        adj[:, 1:] |= reached[:, :-1]
        adj[:, :-1] |= reached[:, 1:]
        reached |= adj & frozen
    return reached


@numba.njit
def random_board(
    shape: tuple[int, int], p: float, contains: Optional[tuple[int, int]] = None
) -> npt.NDArray[np.bool_]:
    connected = np.zeros(shape, dtype=np.bool_)
    start = sample_position_in(connected) if contains is None else contains
    connected[start] = True
    frozen = np.random.random_sample(shape) < p
    connected |= reachable_from(connected, frozen)
    while connected.sum() < p * connected.size:
        extension = sample_position_in(~(connected + frozen))
        frozen[extension] = True
        connected |= reachable_from(connected, frozen + connected)
    return connected


def generate_map(
    shape=(8, 8),
    p=0.8,
    start_pos: tuple[int, int] | None = (0, 0),
    goal_pos: tuple[int, int] | None = (-1, -1),
    multi_start=False,
    mirror=False,
):
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    if start_pos and goal_pos:
        start_pos = start_pos[0] % shape[0], start_pos[1] % shape[1]
        goal_pos = goal_pos[0] % shape[0], goal_pos[1] % shape[1]
        connected = random_board(shape, p, contains=goal_pos)
        while not connected[start_pos]:
            connected = random_board(shape, p, contains=goal_pos)
    else:
        connected = random_board(shape, p, contains=(goal_pos or start_pos))
        start_pos = start_pos or sample_position_in(connected)
        while not goal_pos or goal_pos == start_pos:
            goal_pos = sample_position_in(connected)

    desc = np.empty(shape, dtype="U1")
    desc[connected] = b"F"
    desc[~connected] = b"H"
    if multi_start:
        desc[connected] = "S"
    else:
        desc[start_pos] = "S"
    desc[goal_pos] = "G"
    desc = ["".join(row) for row in desc]
    if mirror:
        desc = [row[::-1] + row[shape[0] % 2 :] for row in desc[::-1] + desc[shape[1] % 2 :]]
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
    width, height = tuple(int(c - 0.4 * s) for s, c in zip(pixels_in_square, pixels_in_cell))
    for x in range(grid_shape[0] // cell_shape[0]):
        for y in range(grid_shape[1] // cell_shape[1]):
            position = tuple(p * c + o for p, c, o in zip((x, y), pixels_in_cell, offset))
            plt.gca().add_patch(
                plt.Rectangle(position, width, height, fc="red", alpha=alpha)  # type: ignore
            )


def plot_trajectory(trajectory: list[ObsType] | list[int], env):
    if trajectory:
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


def plot_qval_heatmap(policy, all_obs_list, env, cmap="RdYlGn", vmin=-1, vmax=1, **kwargs):
    obs_list, valid_mask = all_obs_list
    best_qvals, actions = get_qval(policy, obs_list)
    best_qvals[~np.array(valid_mask, dtype=bool)] = np.nan
    best_qvals = np.array(best_qvals).reshape(env.unwrapped.desc.shape)

    cmap = mpl.colormaps.get_cmap(cmap)  # type: ignore
    cmap.set_bad(color="aqua")
    plt.imshow(best_qvals, cmap=cmap, vmin=-1, vmax=1, **kwargs)
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


def plot_all_abstractions(mango, trajectory=[]):
    env = mango.environment.environment
    plt.figure(figsize=(3 * len(mango.abstract_layers) + 3, 3))
    plt.subplot(1, len(mango.abstract_layers) + 1, 1)
    plt.title(f"Environment")
    plt.imshow(env.render())  # type: ignore
    plot_trajectory(trajectory, env)
    plt.xticks([])
    plt.yticks([])
    for col, layer in enumerate(mango.abstract_layers, start=1):
        plt.subplot(1, len(mango.abstract_layers) + 1, col + 1)
        plt.title(f"Layer {col+1} Abstraction")
        plt.imshow(env.render())  # type: ignore
        plot_trajectory(trajectory, env)
        plot_grid(env, layer.abs_actions.cell_shape)
        plt.xticks([])
        plt.yticks([])


def plot_all_qvals(mango, trajectory=[], **kwargs):
    env = mango.environment.environment
    plt.figure(figsize=(4 * len(Grid2dActions) + 3, 3 * len(mango.abstract_layers)))
    n_rows, n_cols = len(mango.abstract_layers), len(Grid2dActions) + 1
    for row, layer in enumerate(mango.abstract_layers):
        plt.subplot(n_rows, n_cols, row * (len(Grid2dActions) + 1) + 1)
        plt.title(f"Layer {row+1} Abstraction")
        plt.imshow(env.render())  # type: ignore
        plot_trajectory(trajectory, env)
        plot_grid(env, layer.abs_actions.cell_shape)
        plt.xticks([])
        plt.yticks([])
        for col, action in enumerate(Grid2dActions, start=2):
            plt.subplot(n_rows, n_cols, row * (len(Grid2dActions) + 1) + col)
            plt.title(f"Qvals AbsAction {action.name}")
            policy = layer.policy.policies[ActType(action)]
            all_obs_list = all_observations(env, layer.abs_actions.mask)
            plot_qval_heatmap(policy, all_obs_list, env, **kwargs)
            plt.xticks([])
            plt.yticks([])
