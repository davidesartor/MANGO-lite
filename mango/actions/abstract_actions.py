from dataclasses import dataclass, field
from enum import IntEnum
import random
from typing import ClassVar, Optional, Protocol
import numpy as np
from ..utils import ObsType, ActType
from .. import spaces


class AbstractActions(Protocol):
    action_space: spaces.Discrete

    def mask(self, obs: ObsType) -> ObsType:
        return obs

    def beta(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> tuple[bool, bool]:
        ...

    def compatibility(
        self, action: ActType, start_obs: ObsType, next_obs: ObsType
    ) -> float:
        ...


class Grid2dActions(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


@dataclass(eq=False, slots=True, repr=True)
class Grid2dMovement(AbstractActions):
    cell_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    agent_channel: Optional[int] = None
    p_termination: float = 0.1
    reward: float = 1.0
    action_space: ClassVar[spaces.Discrete] = spaces.Discrete(4)

    def obs2coord(self, obs: ObsType) -> tuple[int, int]:
        if self.agent_channel is None:
            idx = obs[0] * self.grid_shape[1] + obs[1]
        else:
            idx = np.argmax(obs[self.agent_channel, :, :])
        y, x = idx // self.grid_shape[1], idx % self.grid_shape[1]
        return int(y // self.cell_shape[0]), int(x // self.cell_shape[1])

    def mask(self, obs: ObsType) -> ObsType:
        if self.agent_channel is None:
            return obs

        padded_obs = np.zeros((obs.shape[0], obs.shape[1] + 2, obs.shape[2] + 2))
        padded_obs[:, 1:-1, 1:-1] = obs

        # add a channel with 0=padd, 1=valid
        # padded_obs = np.zeros((obs.shape[0] + 1, obs.shape[1] + 2, obs.shape[2] + 2))
        # padded_obs[:-1, 1:-1, 1:-1] = obs
        # padded_obs[-1, 1:-1, 1:-1] = 1

        y, x = self.obs2coord(obs)
        y_min = y * self.cell_shape[0]
        y_max = y_min + self.cell_shape[0] + 2
        x_min = x * self.cell_shape[1]
        x_max = x_min + self.cell_shape[1] + 2
        return ObsType(padded_obs[:, y_min:y_max, x_min:x_max])

    def beta(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> tuple[bool, bool]:
        start_y, start_x = self.obs2coord(start_obs)
        next_y, next_x = self.obs2coord(next_obs)
        if start_y != next_y or start_x != next_x:
            return True, False
        return False, random.random() < self.p_termination

    def compatibility(
        self, action: ActType, start_obs: ObsType, next_obs: ObsType
    ) -> float:
        act2delta = {
            Grid2dActions.LEFT: (0, -1),
            Grid2dActions.DOWN: (1, 0),
            Grid2dActions.RIGHT: (0, 1),
            Grid2dActions.UP: (-1, 0),
        }
        start_y, start_x = self.obs2coord(start_obs)
        next_y, next_x = self.obs2coord(next_obs)
        delta_y, delta_x = act2delta[Grid2dActions(action)]
        next_y_expected, next_x_expected = start_y + delta_y, start_x + delta_x
        next_y_expected = max(0, min(self.grid_shape[0]//self.cell_shape[0] - 1, next_y_expected))
        next_x_expected = max(0, min(self.grid_shape[1]//self.cell_shape[1] - 1, next_x_expected))

        if next_y == next_y_expected and next_x == next_x_expected:
            return self.reward
        elif delta_y == 0 and delta_x == 0:
            return 0.0
        else:
            return -self.reward
