from functools import partial
from typing import Callable, Sequence
from flax import linen as nn, struct
import jax

from frozen_lake import FrozenLake, EnvState, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    reward: float
    next_obs: ObsType
    done: bool
    info: dict


class ConvNet(nn.Module):
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x):
        for ch in self.hidden:
            x = nn.Conv(ch, (3, 3))(x)
            x = nn.celu(x)
            x = nn.Conv(ch, (2, 2), strides=(2, 2))(x)
            x = nn.LayerNorm()(x)
        x = x.flatten()
        x = nn.Dense(features=self.out)(x)
        return x
