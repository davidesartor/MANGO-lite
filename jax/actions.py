from functools import partial
import jax
import jax.numpy as jnp

from utils import Transition


def grid_coord(obs, cell_size):
    row, col, cha = obs.shape
    agent_idx = obs[:, :, 0].argmax()
    coord = jnp.array(divmod(agent_idx, col))
    return (coord / cell_size).astype(int)


def beta_fn(cell_sizes: jax.Array, transition: Transition) -> tuple[bool, jax.Array]:
    def single_layer(cell_size, obs, next_obs, action):
        cell_start = grid_coord(obs, cell_size)
        cell_end = grid_coord(next_obs, cell_size)

        rewards = [
            cell_end[1] - cell_start[1] == -1,  # left
            cell_end[0] - cell_start[0] == 1,  # down
            cell_end[1] - cell_start[1] == 1,  # right
            cell_end[0] - cell_start[0] == -1,  # up
            transition.reward,  # goal
        ]
        rewards = jnp.array(rewards, dtype=float).at[:-1].add(-transition.done.astype(float))
        rewards = 2 * rewards - rewards.max()
        stop_cond = (cell_start != cell_end).any()  # |(action == 4)
        return stop_cond, rewards

    return jax.vmap(single_layer, in_axes=(0, None, None, 0))(
        cell_sizes, transition.obs, transition.next_obs, transition.action
    )
