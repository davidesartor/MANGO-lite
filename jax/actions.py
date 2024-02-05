from functools import partial
import jax
import jax.numpy as jnp

from utils import Transition


def grid_coord(obs, cell_size):
    row, col, cha = obs.shape
    agent_idx = obs[:, :, 0].argmax()
    coord = jnp.array(divmod(agent_idx, col))
    return (coord / cell_size).astype(int)


@partial(jax.vmap, in_axes=(0, None))
def beta_fn(cell_size: tuple[int, int], transition: Transition) -> tuple[bool, jax.Array]:
    cell_start = grid_coord(transition.obs, cell_size)
    cell_end = grid_coord(transition.next_obs, cell_size)

    rewards = [
        cell_end[1] - cell_start[1] == -1,  # left
        cell_end[0] - cell_start[0] == 1,  # down
        cell_end[1] - cell_start[1] == 1,  # right
        cell_end[0] - cell_start[0] == -1,  # up
        transition.reward,  # goal
    ]
    rewards = jnp.array(rewards, dtype=float)
    rewards = rewards + 0.25 * (transition.action == 4).astype(float)
    rewards = 2 * rewards - rewards.max()  # if one action succeds, other fail

    stop_cond = (transition.action == 4) | (cell_start != cell_end).any()
    return stop_cond, rewards
