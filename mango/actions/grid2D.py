from dataclasses import dataclass
from enum import IntEnum
import random
from typing import ClassVar, Optional
import numpy as np
from mango.protocols import AbstractActions, ObsType, ActType, Transition
from mango import spaces


@dataclass(eq=False, slots=True, frozen=True, repr=True)
class GridMovement:
    cell_shape: tuple[int, int]
    target_delta: tuple[int, int]

    def abstract(self, obs: ObsType) -> tuple[int, int]:
        # assume obs has shape (C, Y, X) and the agent pos in 1hot encoded in channel 0
        idx = int(np.argmax(obs[0, :, :]))
        y, x = divmod(idx, obs.shape[1])
        return y // self.cell_shape[0], x // self.cell_shape[1]

    def beta(self, trajectory: list[Transition]) -> bool:
        start = self.abstract(trajectory[0].start_obs)
        end = self.abstract(trajectory[-1].next_obs)
        return start != end

    def reward(self, trajectory: list[Transition]) -> float:
        start = self.abstract(trajectory[0].start_obs)
        end = self.abstract(trajectory[-1].next_obs)
        target = tuple(s + d for s, d in zip(start, self.target_delta))

        if start == target:
            return -0.0  # no action, no reward

        if end == target:
            return 0.5 if transition.terminated else 1.0
        else:
            return -1.0

    def mask(self, obs: ObsType) -> ObsType:
        padded_shape = (obs.shape[0] + 1, obs.shape[1] + 2, obs.shape[2] + 2)
        padded_obs = np.zeros_like(obs, shape=padded_shape)
        padded_obs[:-1, 1:-1, 1:-1] = obs
        padded_obs[-1, 1:-1, 1:-1] = 1

        y, x = self.abstract(obs)
        y_min_padd = y * self.cell_shape[0] + 1 + min(0, self.d_y)
        y_max_padd = y_min_padd + self.cell_shape[0] + abs(self.d_y)
        x_min_padd = x * self.cell_shape[1] + 1 + min(0, self.d_x)
        x_max_padd = x_min_padd + self.cell_shape[1] + abs(self.d_x)
        masked_obs = padded_obs[:, y_min_padd:y_max_padd, x_min_padd:x_max_padd]
        return masked_obs


class MoveLeft(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=0, d_x=-1)


class MoveDown(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=1, d_x=0)


class MoveRight(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=0, d_x=1)


class MoveUp(GridMovement):
    def __init__(self, cell_shape: tuple[int, int]):
        super().__init__(cell_shape, d_y=-1, d_x=0)
