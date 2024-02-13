from typing import Callable
import jax
import jax.numpy as jnp
from utils import Transition, ObsType


@jax.jit
def grid_coord(obs: ObsType, cell_size: jax.Array) -> jax.Array:
    row, col, cha = obs.shape
    agent_idx = obs[:, :, 0].argmax()
    coord = jnp.array(divmod(agent_idx, col))
    return coord // cell_size


def get_beta_fn(cell_size: tuple[int, int]) -> Callable[[Transition], bool]:
    @jax.jit
    def beta_fn(transition: Transition) -> bool:
        cell_start = grid_coord(transition.obs, jnp.array(cell_size))
        cell_end = grid_coord(transition.next_obs, jnp.array(cell_size))
        stop_cond = (cell_start != cell_end).any() | transition.done | (transition.action == 4)
        return stop_cond

    return beta_fn


def get_reward_fn(cell_size: tuple[int, int]) -> Callable[[Transition], jax.Array]:
    @jax.jit
    def reward_fn(transition: Transition) -> jax.Array:
        cell_start = grid_coord(transition.obs, jnp.array(cell_size))
        cell_end = grid_coord(transition.next_obs, jnp.array(cell_size))
        dy, dx = (cell_end - cell_start).astype(float)
        reward = jnp.array((-dx, dy, dx, -dy, transition.reward))
        reward = jax.lax.select(transition.done, reward.at[:-1].set(0.0), reward)
        reward = jax.lax.select(transition.action == 4, 0.5 * jnp.ones_like(reward), reward)
        return reward

    return reward_fn
