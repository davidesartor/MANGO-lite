from typing import Any, Protocol, SupportsFloat
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
        fail_on_out_of_bounds: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        if desc is None and map_name == "RANDOM":
            desc = generate_map(**kwargs, seed=seed)
            if fail_on_out_of_bounds:
                desc = ["H" + row + "H" for row in desc]
                desc = ["H" * (len(desc[0]))] + desc + ["H" * (len(desc[0]))]

        super().__init__(render_mode, desc, map_name, is_slippery)
        self.fail_on_out_of_bounds = fail_on_out_of_bounds

    def render(self) -> npt.NDArray[np.uint8]:
        rendered = super().render()
        if self.render_mode == "rgb_array":
            rendered = rendered[: self.cell_size[1] * self.nrow, : self.cell_size[0] * self.ncol]  # type: ignore
        return rendered  # type: ignore

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(4)

    @action_space.setter
    def action_space(self, value: spaces.Discrete) -> None:
        # needed because for some reason the original frozenlake sets it as instance attribute
        pass


class FrozenLakeWrapper(gym.Wrapper, CustomFrozenLakeEnv):
    @property
    def unwrapped(self) -> CustomFrozenLakeEnv:
        return super().unwrapped  # type: ignore

    def observation_inv(self, obs: Any) -> int:
        return obs

    @property
    def action_space(self) -> spaces.Discrete:
        # needed for type hinting
        return spaces.Discrete(4)


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
        super().__init__(env)
        self.one_hot = one_hot

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        y, x = divmod(int(self.unwrapped.s), self.unwrapped.ncol)
        if not self.one_hot:
            obs = np.array([y, x], dtype=np.uint8)
            return obs
        one_hot = np.zeros((1, self.unwrapped.nrow, self.unwrapped.ncol), dtype=np.uint8)
        one_hot[0, y, x] = 1
        return one_hot

    def observation_inv(self, obs: npt.NDArray[np.uint8]) -> int:
        y, x = divmod(int(np.argmax(obs[0])), obs.shape[1]) if self.one_hot else obs
        return int(y * self.unwrapped.ncol + x)


class TensorObservation(FrozenLakeWrapper, gym.ObservationWrapper):
    char2int = {b"S": 1, b"F": 1, b"H": 2, b"G": 3}.get

    def __init__(self, env: CustomFrozenLakeEnv, one_hot=False):
        super().__init__(env)
        self.one_hot = one_hot

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
        super().__init__(env)
        map_shape = (self.unwrapped.nrow, self.unwrapped.ncol)
        if self.unwrapped.fail_on_out_of_bounds:
            map_shape = (map_shape[0] - 2, map_shape[1] - 2)

    def observation(self, observation: int) -> npt.NDArray[np.uint8]:
        render: npt.NDArray[np.uint8] = self.unwrapped._render_gui(mode="rgb_array")  # type: ignore
        return render




