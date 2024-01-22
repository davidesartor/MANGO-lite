from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp


@jax.jit
def connected_components(frozen: jax.Array) -> jax.Array:
    """get a boolean mask (1=frozen, 0=lake) and return a tensor of the same shape
    with the indices of the connected component each cell belongs to (0 if it is a lake)"""

    class Components(NamedTuple):
        last: jax.Array
        prev: jax.Array

    def spread(components: Components) -> Components:
        directional_spreads = [
            jnp.pad(components.last, ((1, 1), (1, 1))),
            jnp.pad(components.last, ((0, 2), (1, 1))),
            jnp.pad(components.last, ((2, 0), (1, 1))),
            jnp.pad(components.last, ((1, 1), (0, 2))),
            jnp.pad(components.last, ((1, 1), (2, 0))),
        ]
        spread = jnp.stack(directional_spreads).max(axis=0)[1:-1, 1:-1]
        spread = spread * frozen
        return Components(last=spread, prev=components.last)

    # expand untill each connected component has an unique id
    components = jax.lax.while_loop(
        cond_fun=lambda components: (components.prev != components.last).any(),
        body_fun=lambda components: spread(components),
        init_val=Components(
            last=jnp.arange(1, frozen.size + 1).reshape(frozen.shape),
            prev=frozen.astype(jnp.int32),
        ),
    )

    # shift the component ids to be contiguous (1, 2, 3, ..., n)
    components = jax.lax.while_loop(
        cond_fun=lambda x: x.min() < 0,
        body_fun=lambda x: jnp.where(x == x.min(), x.max() + 1, x),
        init_val=-components.last,
    )
    return components


@partial(jax.jit, static_argnames=("shape",))
def generate_frozen_chunk(rng_key, shape, p):
    rows, cols = shape
    rng_key, sub_key = jax.random.split(rng_key)
    map = jax.random.uniform(sub_key, shape) < p

    if rows == cols == 1:
        return map

    def while_cond(iter_key_and_map):
        iter, rng_key, map = iter_key_and_map
        return connected_components(map).max() != 1

    def while_body(iter_key_and_map):
        iter, rng_key, map = iter_key_and_map
        rng_key, *sub_keys = jax.random.split(rng_key, 5)
        corners = [(0, 0), (0, cols // 2), (rows // 2, 0), (rows // 2, cols // 2)]
        for key, corner in zip(sub_keys, corners):
            chunk = generate_frozen_chunk(key, (rows // 2, cols // 2), p)
            map = jax.lax.dynamic_update_slice(map, chunk, corner)
        return iter + 1, rng_key, map

    iter, rng_key, map = jax.lax.while_loop(while_cond, while_body, (0, rng_key, map))
    return map


def get_preset_map(rows: int, cols: int) -> jax.Array:
    if rows == cols == 4:
        return 1 - jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
    elif rows == cols == 8:
        return 1 - jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    else:
        raise ValueError(f"no preset map for {rows}x{cols}")


class FrozenLake:
    def __init__(
        self,
        shape: tuple[int, int],
        frozen_prob: Optional[float] = None,
        rng_key: Optional[jax.Array] = None,
    ):
        if rng_key is None:
            self.frozen = get_preset_map(*shape)
            self.agent_start = jnp.array([[0, 0]])
            self.goal_start = jnp.array([[-1, -1]])
            self.rng_key = jax.random.PRNGKey(0)
        else:
            self.rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
            if frozen_prob is None:
                p = jax.random.uniform(subkey1, minval=0.5, maxval=0.8)
            else:
                p = jnp.array(frozen_prob)
            self.frozen = generate_frozen_chunk(subkey2, shape, p)
            self.agent_start = jnp.indices(shape)[:, self.frozen == 1].T
            self.goal_start = jnp.indices(shape)[:, self.frozen == 1].T

    def step(self, action: jax.Array):
        assert action in [0, 1, 2, 3]
        LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
        delta = jnp.select(
            [action == LEFT, action == DOWN, action == RIGHT, action == UP],
            [jnp.array([0, -1]), jnp.array([1, 0]), jnp.array([0, 1]), jnp.array([-1, 0])],
        )
        new_pos = jnp.clip(self.agent_pos + delta, 0, jnp.array(self.frozen.shape))
        print(new_pos)
        self.agent_pos = new_pos

    def reset(self):
        self.rng_key, subkey1, subkey2 = jax.random.split(self.rng_key, 3)
        self.agent_pos = jax.random.choice(subkey1, self.agent_start)
        self.goal_pos = jax.random.choice(subkey2, self.goal_start)
        return self.agent_pos, {}

    def play(self):
        self.reset()
        self.render()
        while True:
            action = jnp.array({"w": 0, "a": 1, "s": 2, "d": 3}.get(input()))
            if action is None:
                break
            self.step(action)
            self.render()

    def render(self):
        import matplotlib.pyplot as plt

        rows, cols = self.frozen.shape
        plt.figure(figsize=(cols, rows), dpi=60)
        plt.xticks([])
        plt.yticks([])
        # plt the map
        plt.imshow(1 - self.frozen, cmap="Blues", vmin=-0.1, vmax=3)
        # plt the frozen
        y, x = jnp.where(self.frozen == 0)
        plt.scatter(x, y, marker="o", s=3000, c="snow", edgecolors="white")
        # plt the holes
        y, x = jnp.where(self.frozen == 0)
        plt.scatter(x, y, marker="o", s=3000, c="tab:blue", edgecolors="white")
        # plt the goal
        y, x = self.goal_pos
        plt.scatter(x, y, marker="*", s=800, c="orange", edgecolors="black")
        # plt the agent
        y, x = self.agent_pos
        plt.scatter(x, y + 0.15, marker="o", s=400, c="pink", edgecolors="black")
        plt.scatter(x, y - 0.15, marker="^", s=400, c="green", edgecolors="black")
        plt.show()
