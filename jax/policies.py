from functools import partial
from typing import Sequence
import flax.linen as nn
import jax


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

    @nn.compact
    def __call__(self, rng_key, obs):
        eps = self.param("epsilon", nn.initializers.ones, ())
        qvals = ConvNet(self.hidden, self.out)(obs)
        return jax.lax.cond(
            jax.random.uniform(rng_key) < eps,
            lambda: jax.random.randint(rng_key, (), 0, self.out),
            lambda: jax.numpy.argmax(qvals),
        )
