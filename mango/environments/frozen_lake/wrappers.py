from typing import Any, Protocol
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
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
        fail_on_out_of_bounds: bool = True,
        seed: int | None = None,
        **kwargs,
    ):
        if desc is None and map_name == "RANDOM":
            desc = generate_map(**kwargs, seed=seed)
            if fail_on_out_of_bounds:
                desc = ["H" + row + "H" for row in desc]
                desc = ["H" * (len(desc[0]))] + desc + ["H" * (len(desc[0]))]

        super().__init__(render_mode, desc, map_name, is_slippery)
        self.action_space = spaces.Discrete(4)
        self.fail_on_out_of_bounds = fail_on_out_of_bounds

    def render(self) -> npt.NDArray[np.uint8]:
        rendered = super().render()
        if self.render_mode == "rgb_array":
            rendered = rendered[: self.cell_size[1] * self.nrow, : self.cell_size[0] * self.ncol]  # type: ignore
        # if self.fail_on_out_of_bounds:
        #     rendered = rendered[self.cell_size[1] : -self.cell_size[1], self.cell_size[0] : -self.cell_size[0]]  # type: ignore
        return rendered  # type: ignore


class FrozenLakeWrapper(gym.Wrapper, CustomFrozenLakeEnv):
    @property
    def unwrapped(self) -> CustomFrozenLakeEnv:
        return super().unwrapped  # type: ignore

    def observation_inv(self, obs: Any) -> int:
        return obs


class ReInitOnReset(FrozenLakeWrapper):
    def __init__(self, env: CustomFrozenLakeEnv, **init_kwargs):
        super().__init__(env)
        self.init_kwargs = init_kwargs
        self.n_resets = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        init_kwargs = self.init_kwargs.copy()
        if seed is not None:
            init_kwargs.update(seed=seed)
        elif "seed" in init_kwargs:
            init_kwargs["seed"] = abs(hash((init_kwargs["seed"], self.n_resets)))
        self.env.__init__(**init_kwargs)
        self.n_resets += 1
        return self.env.reset(seed=seed, options=options)


class CoordinateObservation(FrozenLakeWrapper, gym.ObservationWrapper):
    def __init__(self, env: CustomFrozenLakeEnv, one_hot: bool = False):
        super().__init__(env)  # type: ignore
        self.one_hot = one_hot
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)
        if self.unwrapped.fail_on_out_of_bounds:
            map_shape = (map_shape[0] - 2, map_shape[1] - 2)
        self.observation_space = (
            spaces.Box(low=0, high=max(map_shape), shape=(2,), dtype=np.uint8)
            if not one_hot
            else spaces.Box(low=0, high=1, shape=(1, *map_shape), dtype=np.uint8)
        )

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        y, x = divmod(self.unwrapped.s, self.unwrapped.ncol)  # type: ignore
        if not self.one_hot:
            if self.fail_on_out_of_bounds:
                y, x = y - 1, x - 1
            obs = np.array([y, x], dtype=np.uint8)
            return obs
        one_hot = np.zeros((1, self.unwrapped.nrow, self.unwrapped.ncol), dtype=np.uint8)  # type: ignore
        one_hot[0, y, x] = 1
        return one_hot

    def observation_inv(self, obs: npt.NDArray[np.uint8]) -> int:
        y, x = np.unravel_index(np.argmax(obs[0]), obs.shape[1:]) if self.one_hot else obs
        return int(y * self.unwrapped.ncol + x)  # type: ignore


class TensorObservation(FrozenLakeWrapper, gym.ObservationWrapper):
    char2int = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}.get

    def __init__(self, env: CustomFrozenLakeEnv, one_hot=False):
        super().__init__(env)  # type: ignore
        self.one_hot = one_hot
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)
        if self.unwrapped.fail_on_out_of_bounds:
            map_shape = (map_shape[0] - 2, map_shape[1] - 2)
        self.observation_space = (
            spaces.Box(low=0, high=1, shape=(4, *map_shape), dtype=np.float32)
            if one_hot
            else spaces.Box(low=0, high=3, shape=(1, *map_shape), dtype=np.float32)
        )

    def observation(self, observation: int) -> torch.Tensor:
        map = [[self.char2int(el) for el in list(row)] for row in self.unwrapped.desc]
        row, col = (
            int(self.unwrapped.s) // self.unwrapped.ncol,
            int(self.unwrapped.s) % self.unwrapped.ncol,
        )
        map[row][col] = 0
        map = np.array(map, dtype=np.uint8)
        if not self.one_hot:
            return torch.as_tensor(map[None, :, :], dtype=torch.get_default_dtype())
        one_hot_map = np.zeros((4, *map.shape), dtype=np.uint8)
        Y, X = np.indices(map.shape)
        one_hot_map[map, Y, X] = 1
        one_hot_map = one_hot_map[[0, 2, 3], :, :]  # remove the 1="F"|"S" channel
        return torch.as_tensor(one_hot_map, dtype=torch.get_default_dtype())

    def observation_inv(self, obs: torch.Tensor) -> int:
        agent_idx = torch.argmax(obs[0]) if self.one_hot else torch.argmin(obs[0])
        y, x = divmod(agent_idx.item(), obs.shape[1])
        return int(y) * self.unwrapped.ncol + int(x)


class RenderObservation(FrozenLakeWrapper, gym.ObservationWrapper):
    def __init__(self, env: CustomFrozenLakeEnv):
        super().__init__(env)  # type: ignore
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)
        if self.unwrapped.fail_on_out_of_bounds:
            map_shape = (map_shape[0] - 2, map_shape[1] - 2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, *map_shape), dtype=np.uint8)

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        render: npt.NDArray[np.uint8] = self.unwrapped._render_gui(mode="rgb_array")  # type: ignore
        return render
