from typing import Any, Sequence
import flax.linen as nn
from flax import struct
import jax


PolicyParams = Any


class RandomPolicy(nn.Module):
    out: int

    @nn.compact
    def __call__(self, rng_key, obs):
        return jax.random.randint(rng_key, (), 0, self.out)


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


class DQNet(nn.Module):
    hidden: Sequence[int]
    out: int

    def setup(self):
        self.epsilon = self.param("epsilon", nn.initializers.constant(1.0), ())
        self.qnet = ConvNet(self.hidden, self.out)

    def __call__(self, rng_key, obs):
        qvals = self.qval(obs)
        rng_eps, rng_action = jax.random.split(rng_key)
        return jax.lax.select(
            jax.random.uniform(rng_eps) < self.epsilon,
            jax.random.randint(rng_action, (), 0, self.out),
            jax.numpy.argmax(qvals),
        )

    def qval(self, x: jax.Array) -> jax.Array:
        return self.qnet(x)
