from typing import Any, Sequence
from flax import linen as nn
import flax
import jax

from frozen_lake import ActType, ObsType, RNGKey


class ConvNet(nn.Module):
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x):
        a = self.param("a", nn.initializers.ones, (1, 1))
        for ch in self.hidden:
            x = nn.Conv(ch, (3, 3))(x)
            x = nn.celu(x)
            x = nn.Conv(ch, (2, 2), strides=(2, 2))(x)
            x = nn.LayerNorm()(x)
        x = x.flatten()
        x = nn.Dense(features=self.out)(x)
        return x


PolicyParams = flax.core.scope.VariableDict


class EpsilonGreedyPolicy(nn.Module):
    net: type[nn.Module]
    kwargs: dict[str, Any]

    def setup(self):
        self.epsilon = self.param("epsilon", nn.initializers.ones, ())
        self.qnet = self.net(**self.kwargs)

    def __call__(self, obs: ObsType) -> jax.Array:
        return self.get_qval(obs)

    def get_action(self, rng_key: RNGKey, obs: ObsType) -> ActType:
        rng_eps, rng_action = jax.random.split(rng_key)
        qval = self.get_qval(obs)
        return jax.lax.cond(
            jax.random.uniform(rng_eps) < self.epsilon,
            lambda: jax.random.randint(rng_action, shape=(), minval=0, maxval=qval.size),
            lambda: qval.argmax(),
        )

    def get_qval(self, obs: ObsType) -> jax.Array:
        return self.qnet(obs)
