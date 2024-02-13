import jax
import jax.numpy as jnp
from utils import Transition


def grid_coord(obs, cell_size):
    row, col, cha = obs.shape
    agent_idx = obs[:, :, 0].argmax()
    coord = jnp.array(divmod(agent_idx, col))
    return (coord / cell_size).astype(int)


def beta_fn(cell_size, transition: Transition) -> bool:
    cell_start = grid_coord(transition.obs, cell_size)
    cell_end = grid_coord(transition.next_obs, cell_size)
    stop_cond = (cell_start != cell_end).any()
    return stop_cond


def reward_fn(cell_size: tuple[int, int], transition: Transition) -> jax.Array:
    cell_start = grid_coord(transition.obs, cell_size)
    cell_end = grid_coord(transition.next_obs, cell_size)
    dy, dx = cell_end - cell_start
    rewards = jax.lax.select(
        transition.done,
        jnp.array([0.0, 0.0, 0.0, 0.0, transition.reward]),
        jnp.array([dx == -1, dy == 1, dx == 1, dy == -1, transition.reward]),
    )
    rewards = rewards.at[:4].set(2 * rewards[:4] - rewards[:4].max())
    return rewards
