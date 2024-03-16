from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    out: int
    hidden: Sequence[int] = (512,)

    @nn.compact
    def __call__(self, x):
        x = x.flatten()
        for h in self.hidden:
            x = nn.Dense(h)(x)
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
    n_comands: int
    map_shape: tuple[int, int]
    cell_shape: tuple[int, int]
    hidden: Sequence[int] = (512,)

    @nn.compact
    def __call__(self, x):
        agent_pos, goal_pos = x
        x = jnp.zeros((*self.map_shape, 2))

        # agent_pos = agent_pos // (jnp.array(self.cell_shape) // 2)
        # goal_pos = goal_pos // (jnp.array(self.cell_shape) // 2)

        x = x.at[agent_pos[0], agent_pos[1], 0].set(1)
        x = x.at[goal_pos[0], goal_pos[1], 1].set(1)

        MultiMLP = nn.vmap(
            MLP,
            in_axes=None,  # type: ignore
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_comands,
        )
        x = MultiMLP(self.n_actions, self.hidden)(x)
        return x
