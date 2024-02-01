from typing import Sequence
from flax import linen as nn, struct
import jax

from frozen_lake import FrozenLake, EnvState, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    done: bool
    info: dict


def eps_argmax(rng_key, qval, epsilon):
    """Return argmax with probability 1-epsilon, random idx otherwise."""
    rng_eps, rng_action = jax.random.split(rng_key)
    greedy_action = qval.argmax()
    rand_action = jax.random.choice(rng_action, qval.size)
    return jax.lax.select(jax.random.uniform(rng_eps) > epsilon, greedy_action, rand_action)


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
