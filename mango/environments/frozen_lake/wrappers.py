from typing import Any
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv


from .utils import generate_map
from mango import spaces


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

    def render(self):
        rendered = super().render()
        if self.render_mode == "rgb_array":
            rendered = rendered[: self.cell_size[0] * self.nrow, : self.cell_size[1] * self.ncol]  # type: ignore
        return rendered


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

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        y, x = divmod(self.unwrapped.s, self.unwrapped.ncol)  # type: ignore
        if not self.one_hot:
            return np.array([y, x], dtype=np.uint8)
        one_hot = np.zeros((1, self.unwrapped.nrow, self.unwrapped.ncol), dtype=np.uint8)  # type: ignore
        one_hot[0, y, x] = 1
        return one_hot

    def observation_inv(self, obs: npt.NDArray[np.uint8]) -> int:
        y, x = np.unravel_index(np.argmax(obs[0]), obs.shape[1:]) if self.one_hot else obs
        return int(y * self.unwrapped.ncol + x)  # type: ignore


class TensorObservation(gym.ObservationWrapper):
    char2int = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}.get

    def __init__(self, env: FrozenLakeEnv, one_hot: bool = False):
        super().__init__(env)
        self.one_hot = one_hot
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)  # type: ignore
        self.observation_space = (
            gym.spaces.Box(low=0, high=1, shape=(4, *map_shape), dtype=np.uint8)
            if one_hot
            else gym.spaces.Box(low=0, high=3, shape=(1, *map_shape), dtype=np.uint8)
        )

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        map = [[self.char2int(el) for el in list(row)] for row in self.unwrapped.desc]  # type: ignore
        row, col = divmod(self.unwrapped.s, self.unwrapped.ncol)  # type: ignore
        map[row][col] = 0
        map = np.array(map, dtype=np.uint8)
        if not self.one_hot:
            return map[None, :, :]
        one_hot_map = np.zeros((4, *map.shape), dtype=np.uint8)
        Y, X = np.indices(map.shape)
        one_hot_map[map, Y, X] = 1
        one_hot_map = one_hot_map[[0, 2, 3], :, :]  # remove the 1="F"|"S" channel
        return one_hot_map

    def observation_inv(self, obs: npt.NDArray[np.uint8]) -> int:
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

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        render = self.unwrapped._render_gui(mode="rgb_array")  # type: ignore
        return render
