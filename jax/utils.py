from typing import Sequence
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


def eps_argmax(rng_key, qval, epsilon):
    """Return argmax with probability 1-epsilon, random idx otherwise."""
    rng_eps, rng_action = jax.random.split(rng_key)
    return jax.lax.select(
        jax.random.uniform(rng_eps) > epsilon,
        qval.argmax(),
        jax.random.choice(rng_action, qval.size),
    )


def soft_update(params_qnet_targ, params_qnet, tau):
    return jax.tree_map(lambda pt, p: pt * (1 - tau) + p * tau, params_qnet_targ, params_qnet)


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


MultiConvNet = nn.vmap(
    ConvNet,
    in_axes=None,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_size=5,
)
MultiLayerConvNet = nn.vmap(
    MultiConvNet, in_axes=0, variable_axes={"params": 0}, split_rngs={"params": True}
)
