from typing import Callable
from matplotlib import pyplot as plt
import jax.numpy as jnp
import jax
from flax import linen as nn
import optax
from frozen_lake import FrozenLake, EnvState, ObsType, ActType, RNGKey


def plot_qvals(
    env: FrozenLake,
    qnet_apply_fn: Callable[[optax.Params, ObsType], jax.Array],
    qnet_params: optax.Params,
):
    plt.figure(figsize=(4, 3))
    coords = zip(*jnp.indices(env.frozen.shape).reshape(2, -1))
    env_state, obs = env.reset(jax.random.PRNGKey(0))
    env_states = [env_state.replace(agent_pos=(y, x)) for y, x in coords]
    all_obs = [env.get_obs(jax.random.PRNGKey(0), state) for state in env_states]
    qvals = jnp.stack([qnet_apply_fn(qnet_params, obs) for obs in all_obs])
    plt.imshow(qvals.max(axis=-1).reshape(env.frozen.shape), cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar()


def render(env: FrozenLake, state: EnvState):
    rows, cols = env.frozen.shape
    plt.figure(figsize=(cols, rows), dpi=60)
    plt.xticks([])
    plt.yticks([])
    # plt the map
    plt.imshow(1 - env.frozen, cmap="Blues", vmin=-1, vmax=3)
    # plt the frozen
    y, x = jnp.where(env.frozen == 1)
    plt.scatter(x, y, marker="o", s=2500, c="snow", edgecolors="k")
    # plt the frozen
    y, x = jnp.where(env.frozen == 0)
    plt.scatter(x, y, marker="o", s=2500, c="tab:blue", edgecolors="w")

    # plt the goal
    y, x = state.goal_pos if state.goal_pos.ndim == 1 else state.goal_pos[-1]
    plt.scatter(x, y, marker="*", s=800, c="orange", edgecolors="k")

    # plt the agent
    if state.agent_pos.ndim == 1:
        y, x = state.agent_pos
        plt.scatter(x, y + 0.15, marker="o", s=400, c="pink", edgecolors="k")
        plt.scatter(x, y - 0.15, marker="^", s=400, c="green", edgecolors="k")
    else:
        # if superposition, use fequency as alpha
        frequency, _, _ = jnp.histogram2d(
            state.agent_pos[:, 0],
            state.agent_pos[:, 1],
            bins=env.frozen.shape,
            range=[[0, s] for s in env.frozen.shape],
        )
        alpha = 0.2 + 0.6 * frequency / frequency.max()
        y, x = jnp.where(frequency > 0)
        plt.scatter(x, y + 0.15, marker="o", s=400, c="pink", edgecolors="k", alpha=alpha[y, x])  # type: ignore
        plt.scatter(x, y - 0.15, marker="^", s=400, c="green", edgecolors="k", alpha=alpha[y, x])  # type: ignore
