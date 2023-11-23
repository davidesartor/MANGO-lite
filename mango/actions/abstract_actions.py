from typing import Protocol
from ..utils import ObsType, ActType
from .. import spaces


class AbstractActions(Protocol):
    action_space: spaces.Discrete

    def mask(self, obs: ObsType) -> ObsType:
        return obs

    def beta(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> tuple[bool, bool]:
        ...

    def compatibility(self, action: ActType, start_obs: ObsType, next_obs: ObsType) -> float:
        ...
