import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from frozen_lake import EnvState, EnvParams


def render(state: EnvState, env_params: EnvParams):
    rows, cols = env_params.frozen.shape
    plt.figure(figsize=(cols, rows), dpi=60)
    plt.xticks([])
    plt.yticks([])
    # plt the map
    plt.imshow(1 - env_params.frozen, cmap="Blues", vmin=-1, vmax=3)
    # plt the frozen
    y, x = jnp.where(env_params.frozen == 1)
    plt.scatter(x, y, marker="o", s=2500, c="snow", edgecolors="k")
    # plt the frozen
    y, x = jnp.where(env_params.frozen == 0)
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
            bins=env_params.frozen.shape,
        )
        alpha = 0.2 + 0.6 * frequency / frequency.max()
        y, x = jnp.where(frequency > 0)
        plt.scatter(x, y + 0.15, marker="o", s=400, c="pink", edgecolors="k", alpha=alpha[y, x])
        plt.scatter(x, y - 0.15, marker="^", s=400, c="green", edgecolors="k", alpha=alpha[y, x])
