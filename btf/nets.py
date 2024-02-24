import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    out: int
    hidden: int = 512

    @nn.compact
    def __call__(self, x):
        x = x.flatten()
        x = nn.Dense(self.hidden)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out)(x)
        return x


class Qnet(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = MLP(self.n_actions)(x)
        return x


class MultiTaskQnet(nn.Module):
    n_actions: int
    n_comands: int = 1

    @nn.compact
    def __call__(self, x):
        x = jnp.stack([MLP(self.n_actions)(x) for _ in range(self.n_comands)], axis=0)
        return x

        MultiMLP = nn.vmap(
            MLP,
            in_axes=None,  # type: ignore
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_comands,
        )
        x = MultiMLP(self.n_actions)(x)
        return x
