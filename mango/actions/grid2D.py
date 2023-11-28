from dataclasses import dataclass
from enum import IntEnum
import random
from typing import ClassVar, Optional
import numpy as np
from mango.protocols import AbstractActions, ObsType, ActType
from mango import spaces


class Actions(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    def to_delta(self) -> tuple[int, int]:
        return {
            Actions.LEFT: (0, -1),
            Actions.DOWN: (1, 0),
            Actions.RIGHT: (0, 1),
            Actions.UP: (-1, 0),
        }[self]


@dataclass(eq=False, slots=True, frozen=True, repr=True)
class SubGridMovement(AbstractActions):
    cell_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    agent_channel: Optional[int] = None
    invalid_channel: Optional[int] = None
    p_termination: float = 0.1
    reward: float = 1.0

    action_space: ClassVar = spaces.Discrete(4)

    def obs2coord(self, obs: ObsType) -> tuple[int, int]:
        if self.agent_channel is not None:
            idx = np.argmax(obs[self.agent_channel, :, :])
            y, x = idx // self.grid_shape[1], idx % self.grid_shape[1]
        else:
            y, x = obs
        return int(y // self.cell_shape[0]), int(x // self.cell_shape[1])

    def beta(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> tuple[bool, bool]:
        start_y, start_x = self.obs2coord(start_obs)
        next_y, next_x = self.obs2coord(next_obs)
        if start_y != next_y or start_x != next_x:
            return True, False
        return False, random.random() < self.p_termination

    def compatibility(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> float:
        start_y, start_x = self.obs2coord(start_obs)
        next_y, next_x = self.obs2coord(next_obs)
        delta_y, delta_x = Actions.to_delta(Actions(int(action)))
        next_y_expected, next_x_expected = start_y + delta_y, start_x + delta_x

        if next_y == next_y_expected and next_x == next_x_expected:
            return self.reward
        elif next_y == start_y and next_x == start_x:
            return 0.0
        else:
            return -1.0

    def mask(self, comand: ActType, obs: ObsType) -> ObsType:
        if self.agent_channel is None:
            return obs

        if self.invalid_channel is None:
            padded_obs = np.zeros((obs.shape[0] + 1, obs.shape[1] + 2, obs.shape[2] + 2))
            padded_obs[:-1, 1:-1, 1:-1] = obs
            padded_obs[-1, 1:-1, 1:-1] = 1
        else:
            padded_obs = np.zeros((obs.shape[0], obs.shape[1] + 2, obs.shape[2] + 2))
            padded_obs[self.invalid_channel] = 1
            padded_obs[:, 1:-1, 1:-1] = obs

        y, x = self.obs2coord(obs)
        d_y, d_x = Actions.to_delta(Actions(int(comand)))
        y_min_padd = y * self.cell_shape[0] + 1 + min(0, d_y)
        y_max_padd = y_min_padd + self.cell_shape[0] + abs(d_y)
        x_min_padd = x * self.cell_shape[1] + 1 + min(0, d_x)
        x_max_padd = x_min_padd + self.cell_shape[1] + abs(d_x)
        masked_obs = padded_obs[:, y_min_padd:y_max_padd, x_min_padd:x_max_padd]
        return masked_obs
