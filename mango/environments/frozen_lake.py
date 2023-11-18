from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
from .. import spaces
from ..utils import ActType, ObsType


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
        **kwargs
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
        y, x = np.unravel_index(np.argmax(obs[0]), obs.shape[1:]) if self.onehot else obs
        return int(y * self.unwrapped.ncol + x)  # type: ignore

    def all_observations(self) -> list[ObsType]:
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)  # type: ignore
        y_matrix, x_matrix = np.indices(map_shape)
        obs_list = []
        for y, x in zip(y_matrix.flatten(), x_matrix.flatten()):
            if not self.one_hot:
                obs_list.append(ObsType(np.array([y, x])))
            else:
                obs_list.append(np.zeros((1, *map_shape), dtype=np.uint8))
                obs_list[-1][0, y, x] = 1
        return obs_list


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

    def all_observations(self) -> list[ObsType]:
        base_map = [[self.char2int(el) for el in list(row)] for row in self.unwrapped.desc]  # type: ignore
        base_map = np.array(base_map, dtype=np.uint8)

        y_matrix, x_matrix = np.indices(base_map.shape)
        obs_list = []
        for y, x in zip(y_matrix.flatten(), x_matrix.flatten()):
            map = base_map.copy()
            map[y, x] = 0
            if not self.one_hot:
                obs_list.append(ObsType(map[None, :, :]))
            else:
                one_hot_map = np.zeros((4, *map.shape), dtype=np.uint8)
                Y, X = np.indices(map.shape)
                one_hot_map[map, Y, X] = 1
                obs_list.append(ObsType(one_hot_map[[0, 2, 3], :, :]))
        return obs_list


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
