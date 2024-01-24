from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from flax.core.scope import CollectionFilter, FrozenVariableDict, VariableDict
import flax.linen as nn
from flax import struct
from flax.linen.module import Module, RNGSequences
import jax

from frozen_lake import ActType, ObsType, RNGKey


NetParams = Any


class PolicyParams(struct.PyTreeNode):
    net_params: NetParams
    epsilon: float = 0.1


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

    # temp to silence type warnings
    def apply(self, params: NetParams, obs: ObsType) -> jax.Array:
        return super().apply(params, obs)  # type: ignore


def epsilon_greedy(
    policy_params: PolicyParams,
    rng_key: RNGKey,
    obs: ObsType,
    get_qval_fn: Callable[[PolicyParams, ObsType], jax.Array],
) -> ActType:
    rng_eps, rng_action = jax.random.split(rng_key)
    qval = get_qval_fn(policy_params, obs)
    return jax.lax.cond(
        jax.random.uniform(rng_eps) < policy_params.epsilon,
        lambda: jax.random.randint(rng_action, shape=(), minval=0, maxval=qval.size),
        lambda: qval.argmax(),
    )
