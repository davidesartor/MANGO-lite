from functools import partial, wraps
from typing import Any, Callable, Protocol, Sequence
from flax import struct
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from frozen_lake import EnvState, FrozenLake, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
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


def epsilon_greedy_policy(qval_apply_fn: Callable):
    def policy_fn(
        params: optax.Params, rng_key: RNGKey, obs: ObsType, randomness: float
    ) -> ActType:
        rng_eps, rng_action = jax.random.split(rng_key)
        qval = qval_apply_fn(params, obs)
        action = jax.lax.select(
            jax.random.uniform(rng_eps) < randomness,
            jax.random.randint(rng_action, shape=(), minval=0, maxval=qval.size),
            qval.argmax(),
        )
        return action

    return wraps(policy_fn)(jax.jit(policy_fn))
