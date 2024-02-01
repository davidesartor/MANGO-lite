from functools import partial
from typing import ClassVar, Optional
import spaces

import jax
import jax.numpy as jnp
from flax import struct


RNGKey = jax.Array
ObsType = jax.Array
ActType = jax.Array


class EnvState(struct.PyTreeNode):
    agent_pos: jax.Array
    goal_pos: jax.Array


class FrozenLake(struct.PyTreeNode):
    frozen: jax.Array
    agent_start: jax.Array
    goal_start: jax.Array
    action_space: spaces.Space = struct.field(pytree_node=False, default=spaces.Discrete(4))

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "shape"))
    def make(
        cls,
        rng_key: RNGKey,
        shape: tuple[int, int],
        frozen_prob: Optional[float] = None,
        frozen_prob_high: Optional[float] = None,
    ):
        if frozen_prob is None:
            frozen = get_preset_map(shape)
            agent_start = jnp.array([[0, 0]])
            goal_start = jnp.array([[s - 1 for s in shape]])
        else:
            rng_p, rng_gen = jax.random.split(rng_key)
            p_high = frozen_prob_high or frozen_prob
            p = jax.random.uniform(rng_p, minval=frozen_prob, maxval=p_high)
            frozen = generate_frozen_chunk(rng_gen, shape, p)
            agent_start = jnp.indices(shape)[:, frozen].T
            goal_start = jnp.indices(shape)[:, frozen].T
        return cls(frozen, agent_start, goal_start)

    @jax.jit
    def reset(self, rng_key: RNGKey) -> tuple[EnvState, ObsType]:
        rng_agent, rng_goal, rng_obs = jax.random.split(rng_key, 3)
        agent_pos = jax.random.choice(rng_agent, self.agent_start)
        goal_pos = jax.random.choice(rng_goal, self.goal_start)
        state = jax.lax.stop_gradient(EnvState(agent_pos, goal_pos))
        obs = self.get_obs(rng_obs, state)
        return state, obs

    @partial(jax.jit, donate_argnames=("state",))
    def step(
        self, state: EnvState, rng_key: RNGKey, action: ActType
    ) -> tuple[EnvState, ObsType, float, bool, dict]:
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])[action]
        new_agent_pos = jnp.clip(state.agent_pos + delta, 0, jnp.array(self.frozen.shape) - 1)
        state = state.replace(agent_pos=new_agent_pos)
        obs = self.get_obs(rng_key, state)

        reward, done = jax.lax.cond(
            (state.agent_pos == state.goal_pos).all(),
            lambda: (1.0, True),
            lambda: (0.0, ~self.frozen[tuple(state.agent_pos)]),
        )
        return state, obs, reward, done, {}

    @jax.jit
    def get_obs(self, rng_key: RNGKey, state: EnvState) -> ObsType:
        # one-hot encoding of the observation
        obs = jnp.zeros((*self.frozen.shape, 3))
        obs = obs.at[state.agent_pos[0], state.agent_pos[1], 0].set(1)
        obs = obs.at[state.goal_pos[0], state.goal_pos[1], 1].set(1)
        obs = obs.at[:, :, 2].set(~self.frozen)
        return jax.lax.stop_gradient(obs)


@jax.jit
def connected_components(frozen: jax.Array) -> jax.Array:
    """get a boolean mask (1=frozen, 0=lake) and return a tensor of the same shape
    with the indices of the connected component each cell belongs to (0 if it is a lake)"""

    def while_cond(prev_and_curr_components):
        previous, current = prev_and_curr_components
        return (previous != current).any()

    def while_body(prev_and_curr_components):
        previous, current = prev_and_curr_components
        spread = current
        spread = spread.at[1:, :].set(jnp.maximum(spread[1:, :], current[:-1, :]))
        spread = spread.at[:-1, :].set(jnp.maximum(spread[:-1, :], current[1:, :]))
        spread = spread.at[:, 1:].set(jnp.maximum(spread[:, 1:], current[:, :-1]))
        spread = spread.at[:, :-1].set(jnp.maximum(spread[:, :-1], current[:, 1:]))
        spread = spread * frozen
        return current, spread

    # expand untill each connected component has an unique id
    components = jnp.arange(1, frozen.size + 1).reshape(frozen.shape)
    _, components = jax.lax.while_loop(
        cond_fun=while_cond, body_fun=while_body, init_val=(frozen.astype(jnp.int32), components)
    )

    # shift the component ids to be contiguous (1, 2, 3, ..., n)
    components = jax.lax.while_loop(
        cond_fun=lambda x: x.min() < 0,
        body_fun=lambda x: jnp.where(x == x.min(), x.max() + 1, x),
        init_val=-components,
    )
    return components


@partial(jax.jit, static_argnames=("shape",))
def generate_frozen_chunk(rng_key: RNGKey, shape: tuple[int, int], p: jax.Array) -> jax.Array:
    rows, cols = shape
    rng_key, sub_key = jax.random.split(rng_key)
    map = jax.random.uniform(sub_key, shape) < p

    if rows == cols == 1:
        return map

    def while_cond(iter_rng_and_map):
        iter, rng_key, map = iter_rng_and_map
        return connected_components(map).max() != 1

    def while_body(iter_rng_and_map):
        iter, rng_key, map = iter_rng_and_map
        rng_key, *sub_keys = jax.random.split(rng_key, 5)
        corners = [(0, 0), (0, cols // 2), (rows // 2, 0), (rows // 2, cols // 2)]
        for key, corner in zip(sub_keys, corners):
            chunk = generate_frozen_chunk(key, (rows // 2, cols // 2), p)
            map = jax.lax.dynamic_update_slice(map, chunk, corner)
        return iter + 1, rng_key, map

    iter, rng_key, map = jax.lax.while_loop(while_cond, while_body, (0, rng_key, map))
    return map


@partial(jax.jit, static_argnames=("shape"))
def get_preset_map(shape: tuple[int, int]) -> jax.Array:
    rows, cols = shape
    if rows == cols == 2:
        map = ["FH", "FF"]
    elif rows == cols == 4:
        map = ["FFFF", "FHFH", "FFFH", "HFFF"]
    elif rows == cols == 8:
        map = [
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFF",
        ]
    else:
        raise ValueError(f"no preset map for {rows}x{cols}")
    return jnp.array([[c == "F" for c in row] for row in map])
