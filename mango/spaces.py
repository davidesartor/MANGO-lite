import gymnasium as gym
import numpy as np


# eventually mango should  probably be changed to use gym.spaces directly
# but it is convenient to extend some functionality (i.e. __iter__ for Discrete)


class Discrete(gym.spaces.Discrete):
    def __iter__(self):
        return iter(np.arange(self.n))


class Space(gym.spaces.Space):
    pass


class Box(gym.spaces.Box):
    pass
