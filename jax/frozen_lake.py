from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional
import spaces

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EnvParams:
    frozen: jax.Array
    agent_start: jax.Array
    goal_start: jax.Array


@struct.dataclass
class EnvState:
    agent_pos: jax.Array
    goal_pos: jax.Array


EnvObs = jax.Array


@dataclass(eq=False, frozen=True)
class FrozenLake:
    shape: tuple[int, int]
    frozen_prob: Optional[float] = None
    frozen_prob_high: Optional[float] = None

    action_space: spaces.Space = spaces.Discrete(4)

    def init(self, rng_key: jax.Array) -> EnvParams:
        if self.frozen_prob is None:
            frozen = get_preset_map(self.shape)
            agent_start = jnp.array([[0, 0]])
            goal_start = jnp.array([[s - 1 for s in self.shape]])
        else:
            key, key_p, key_gen = jax.random.split(rng_key, 3)
            p_high = self.frozen_prob_high or self.frozen_prob
            p = jax.random.uniform(key_p, minval=self.frozen_prob, maxval=p_high)
            frozen = generate_frozen_chunk(key_gen, self.shape, p)
            agent_start = jnp.indices(self.shape)[:, frozen].T
            goal_start = jnp.indices(self.shape)[:, frozen].T
        return EnvParams(frozen, agent_start, goal_start)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng_key: jax.Array, env_params: EnvParams) -> tuple[EnvObs, EnvState]:
        rng_key, key_agent, key_goal = jax.random.split(rng_key, 3)
        agent_pos = jax.random.choice(key_agent, env_params.agent_start)
        goal_pos = jax.random.choice(key_goal, env_params.goal_start)
        state = EnvState(agent_pos, goal_pos)
        return self.obs(state, env_params), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, rng_key: jax.Array, state: EnvState, action: jax.Array, env_params: EnvParams
    ) -> tuple[EnvObs, EnvState, float, bool, dict]:
        LEFT, DOWN, RIGHT, UP = jnp.array(0), jnp.array(1), jnp.array(2), jnp.array(3)
        delta = jnp.select(
            [action == LEFT, action == DOWN, action == RIGHT, action == UP],
            [jnp.array([0, -1]), jnp.array([1, 0]), jnp.array([0, 1]), jnp.array([-1, 0])],
        )
        new_agent_pos = jnp.clip(state.agent_pos + delta, 0, jnp.array(env_params.frozen.shape) - 1)
        state = EnvState(agent_pos=new_agent_pos, goal_pos=state.goal_pos)

        reward, done = jax.lax.cond(
            (state.agent_pos == state.goal_pos).all(),
            lambda: (1.0, True),
            lambda: (0.0, ~env_params.frozen[tuple(state.agent_pos)]),
        )

        return self.obs(state, env_params), state, reward, done, {}

    def obs(self, state: EnvState, env_params: EnvParams) -> EnvObs:
        # one-hot encoding of the observation
        obs = jnp.zeros((3, *env_params.frozen.shape))
        obs = obs.at[0, state.agent_pos[0], state.agent_pos[1]].set(1)
        obs = obs.at[1, state.goal_pos[0], state.goal_pos[1]].set(1)
        obs = obs.at[2].set(env_params.frozen)
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


@partial(jax.jit, static_argnames=("shape"))
def get_preset_map(shape) -> jax.Array:
    rows, cols = shape
    if rows == cols == 4:
        map = [
            "FFFF",
            "FHFH",
            "FFFH",
            "HFFF",
        ]

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
