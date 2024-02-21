from typing import Callable
import jax
import jax.numpy as jnp
from utils import Transition, ObsType


def grid_coord(obs: ObsType, cell_scale: int) -> jax.Array:
    row, col, cha = obs.shape
    agent_idx = obs[:, :, 0].argmax()
    coord = jnp.array(divmod(agent_idx, col))
    return coord // 2**cell_scale


def beta_fn(cell_scale: int, transition: Transition) -> jnp.bool_:
    cell_start = grid_coord(transition.obs, cell_scale)
    cell_end = grid_coord(transition.next_obs, cell_scale)
    stop_cond = (cell_start != cell_end).any() | transition.done
    return stop_cond


def reward_fn(cell_scale: int, transition: Transition) -> jax.Array:
    cell_start = grid_coord(transition.obs, cell_scale)
    cell_end = grid_coord(transition.next_obs, cell_scale)
    dy, dx = (cell_end - cell_start).astype(float)
    reward = jnp.array((-dx, dy, dx, -dy, transition.reward))
    reward = jax.lax.select(transition.done, reward.at[:-1].set(0.0), reward)
    return reward
