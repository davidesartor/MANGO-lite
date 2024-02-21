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


n_comands = 5
MultiMLP = nn.vmap(
    MLP,
    in_axes=None,  # type: ignore
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_size=n_comands,
)


class InnerQnet(nn.Module):
    n_actions: int
    map_shape: tuple[int, int]

    @nn.compact
    def __call__(self, obs):
        x = jnp.zeros((*self.map_shape, 2))
        x = x.at[obs[0, 0], obs[0, 1], 0].set(1)
        x = x.at[obs[1, 0], obs[1, 1], 1].set(1)
        x = MultiMLP(self.n_actions)(x)
        return x


class OuterQnet(nn.Module):
    n_actions: int
    map_shape: tuple[int, int]

    @nn.compact
    def __call__(self, obs):
        x = jnp.zeros((*self.map_shape, 2))
        x = x.at[obs[0, 0], obs[0, 1], 0].set(1)
        x = x.at[obs[1, 0], obs[1, 1], 1].set(1)
        x = MLP(self.n_actions)(x)
        return x
