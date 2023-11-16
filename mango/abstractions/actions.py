from dataclasses import dataclass, field
import random
from typing import Protocol
import numpy as np
from ..utils import ObsType, ActType
from .. import spaces


class AbstractActions(Protocol):
    action_space: spaces.Discrete

    def beta(self, start_obs: ObsType, next_obs: ObsType) -> bool:
        ...

    def compatibility(
        self, action: ActType, start_obs: ObsType, next_obs: ObsType
    ) -> float:
        ...


@dataclass(eq=False, slots=True, frozen=True)
class GridMovement(AbstractActions):
    cell_shape: tuple[int, int]
    action_space: spaces.Discrete = field(init=False, default=spaces.Discrete(4))

    def beta(self, start_obs: ObsType, next_obs: ObsType) -> bool:
        start_y, start_x = self.obs2yxpos(start_obs)
        next_y, next_x = self.obs2yxpos(next_obs)
        if start_y != next_y or start_x != next_x:
            return True
        return random.random() < 0.1

    def compatibility(
        self, action: ActType, start_obs: ObsType, next_obs: ObsType
    ) -> float:
        start_y, start_x = self.obs2yxpos(start_obs)
        next_y, next_x = self.obs2yxpos(next_obs)
        delta_y, delta_x = next_y - start_y, next_x - start_x

        LEFT, DOWN, RIGHT, UP = ActType(0), ActType(1), ActType(2), ActType(3)
        if delta_y == 0 and delta_x == -1:
            executed = LEFT
        elif delta_y == 1 and delta_x == 0:
            executed = DOWN
        elif delta_y == 0 and delta_x == 1:
            executed = RIGHT
        elif delta_y == -1 and delta_x == 0:
            executed = UP
        else:
            executed = None

        if executed == action:
            return 0.5
        elif delta_y == 0 and delta_x == 0:
            return -0.1
        else:
            return -0.5

    def obs2yxpos(self, obs: ObsType) -> tuple[int, int]:
        y, x = obs
        return int(y // self.cell_shape[0]), int(x // self.cell_shape[1])


@dataclass(eq=False, slots=True, frozen=True)
class GridMovementOneHot(GridMovement):
    channel: int = 0

    def obs2yxpos(self, obs: ObsType) -> tuple[int, int]:
        y, x = np.unravel_index(np.argmax(obs[self.channel, :, :]), obs.shape[1:])
        return int(y // self.cell_shape[0]), int(x // self.cell_shape[1])
