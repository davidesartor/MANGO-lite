import numpy as np
from gymnasium.spaces.space import MaskNDArray
from .utils import ObsType, ActType
import gymnasium as gym


class Discrete(gym.spaces.Discrete):
    def __iter__(self):
        return iter([ActType(elem) for elem in range(self.n)])
