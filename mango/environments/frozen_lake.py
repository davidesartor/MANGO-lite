from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map


def generate_map(size=8, p=0.8, mirror=False, random_start=False):
    desc = generate_random_map(size=size // 2 if mirror else size, p=p)
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
        is_slippery=True,
        **kwargs
    ):
        if map_name == "RANDOM":
            desc = generate_map(**kwargs)
        super().__init__(render_mode, desc, map_name, is_slippery)


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
    def __init__(self, env: FrozenLakeEnv):
        super().__init__(env)
        max_coord = max((self.unwrapped.nrow, self.unwrapped.ncol))  # type: ignore
        self.observation_space = gym.spaces.Box(
            low=0, high=max_coord, shape=(2,), dtype=np.uint8
        )

    def observation(self, observation: int) -> Any:
        y, x = divmod(self.unwrapped.s, self.unwrapped.ncol)  # type: ignore
        return np.array([y, x], dtype=np.uint8)


class TensorObservation(gym.ObservationWrapper):
    char2int = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}.get

    def __init__(self, env: gym.Env, one_hot: bool = False):
        super().__init__(env)
        self.one_hot = one_hot
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)  # type: ignore
        self.observation_space = (
            gym.spaces.Box(low=0, high=1, shape=(*map_shape, 4), dtype=np.uint8)
            if one_hot
            else gym.spaces.Box(low=0, high=3, shape=map_shape, dtype=np.uint8)
        )

    def observation(self, observation: int) -> Any:
        map = [[self.char2int(el) for el in list(row)] for row in self.unwrapped.desc]  # type: ignore
        row, col = divmod(self.unwrapped.s, self.unwrapped.ncol)  # type: ignore
        map[row][col] = 0
        map = np.array(map, dtype=np.uint8)
        if not self.one_hot:
            return map
        one_hot_map = np.zeros((*map.shape, 4), dtype=np.uint8)
        one_hot_map[np.arange(map.shape[0]), np.arange(map.shape[1]), map] = 1
        one_hot_map = one_hot_map[:, :, [0, 2, 3]]  # remove the 1="F"|"S" channel
        return one_hot_map


class RenderObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)  # type: ignore
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*map_shape, 3), dtype=np.uint8
        )

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        render = self.unwrapped._render_gui(mode="rgb_array")  # type: ignore
        return render
