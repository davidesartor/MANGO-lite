from dataclasses import dataclass
from enum import IntEnum
import random
from typing import ClassVar
import numpy as np
from ..utils import ObsType, ActType
from .. import spaces
from .abstract_actions import AbstractActions


class Actions(IntEnum):
    LEFT = ActType(0)
    DOWN = ActType(1)
    RIGHT = ActType(2)
    UP = ActType(3)

    def to_delta(self) -> tuple[int, int]:
        return {
            Actions.LEFT: (0, -1),
            Actions.DOWN: (1, 0),
            Actions.RIGHT: (0, 1),
            Actions.UP: (-1, 0),
        }[self]


@dataclass(eq=False, slots=True, repr=True)
class SubGridMovement(AbstractActions):
    action_space: ClassVar[spaces.Discrete] = spaces.Discrete(4)
    cell_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    p_termination: float = 0.1
    reward: float = 1.0

    def obs2coord(self, obs: ObsType) -> tuple[int, int]:
        y, x = obs
        return int(y // self.cell_shape[0]), int(x // self.cell_shape[1])

    def mask(self, comand: ActType, obs: ObsType) -> ObsType:
        return obs

    def beta(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> tuple[bool, bool]:
        start_y, start_x = self.obs2coord(start_obs)
        next_y, next_x = self.obs2coord(next_obs)
        if start_y != next_y or start_x != next_x:
            return True, False
        return False, random.random() < self.p_termination

    def compatibility(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> float:
        start_y, start_x = self.obs2coord(start_obs)
        next_y, next_x = self.obs2coord(next_obs)
        delta_y, delta_x = Actions.to_delta(Actions(action))
        next_y_expected, next_x_expected = start_y + delta_y, start_x + delta_x

        if next_y == next_y_expected and next_x == next_x_expected:
            return self.reward
        elif next_y == start_y and next_x == start_x:
            return 0.0
        else:
            return -1.0


@dataclass(eq=False, slots=True, repr=True)
class SubGridMovementOnehot(SubGridMovement):
    agent_channel: int = 0
    add_valid_channel: bool = True

    def obs2coord(self, obs: ObsType) -> tuple[int, int]:
        idx = np.argmax(obs[self.agent_channel, :, :])
        y, x = idx // self.grid_shape[1], idx % self.grid_shape[1]
        return int(y // self.cell_shape[0]), int(x // self.cell_shape[1])

    def mask(self, comand: ActType, obs: ObsType) -> ObsType:
        if obs.shape[0] == 1:
            return obs

        if self.add_valid_channel:
            padded_obs = np.zeros((obs.shape[0] + 1, obs.shape[1] + 2, obs.shape[2] + 2))
            padded_obs[:-1, 1:-1, 1:-1] = obs
            padded_obs[-1, 1:-1, 1:-1] = 1
        else:
            padded_obs = np.zeros((obs.shape[0], obs.shape[1] + 2, obs.shape[2] + 2))
            padded_obs[:, 1:-1, 1:-1] = obs

        y, x = self.obs2coord(obs)
        d_y, d_x = Actions.to_delta(Actions(comand))
        y_min_padd = y * self.cell_shape[0] + 1 + min(0, d_y)
        y_max_padd = y_min_padd + self.cell_shape[0] + max(0, d_y)
        x_min_padd = x * self.cell_shape[1] + 1 + min(0, d_x)
        x_max_padd = x_min_padd + self.cell_shape[1] + max(0, d_x)
        masked_obs = padded_obs[:, y_min_padd:y_max_padd, x_min_padd:x_max_padd]
        return masked_obs
