from functools import partial
import jax
import jax.numpy as jnp


def grid_coord(obs, cell_size):
    agent_idx = obs[:, :, 0].argmax()
    coord = jnp.array(divmod(agent_idx, obs.shape[1]))
    return (coord / cell_size).astype(int)


@partial(jax.vmap, in_axes=(0, None))
def reward_fn(cell_size, transition):
    cell_start = grid_coord(transition.obs, cell_size)
    cell_end = grid_coord(transition.next_obs, cell_size)

    rewards = [
        cell_end[1] - cell_start[1] == -1,  # left
        cell_end[0] - cell_start[0] == 1,  # down
        cell_end[1] - cell_start[1] == 1,  # right
        cell_end[0] - cell_start[0] == -1,  # up
        transition.reward == 1,  # goal
    ]
    rewards = jnp.array(rewards, dtype=float)
    return 2 * rewards - rewards.max()


@partial(jax.vmap, in_axes=(0, None, None))
def beta_fn(cell_size, transition):
    cell_start = grid_coord(transition.obs, cell_size)
    cell_end = grid_coord(transition.next_obs, cell_size)
    stop_cond = transition.action == 4 or (cell_start != cell_end).any()
    return jax.lax.select(stop_cond, True, False)
