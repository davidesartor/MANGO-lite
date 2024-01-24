from functools import partial, wraps
from typing import ClassVar, Optional
import spaces

import jax
import jax.numpy as jnp
from flax import struct


class EnvParams(struct.PyTreeNode):
    frozen: jax.Array
    agent_start: jax.Array
    goal_start: jax.Array


class EnvState(struct.PyTreeNode):
    agent_pos: jax.Array
    goal_pos: jax.Array


RNGKey = jax.Array
ObsType = jax.Array
ActType = jax.Array


class FrozenLake:
    action_space: ClassVar[spaces.Space] = spaces.Discrete(4)

    def __init__(
        self,
        shape: tuple[int, int],
        frozen_prob: Optional[float] = None,
        frozen_prob_high: Optional[float] = None,
    ):
        self.shape = shape
        self.frozen_prob = frozen_prob
        self.frozen_prob_high = frozen_prob_high

    @partial(jax.jit, static_argnums=0)
    def init(self, rng_key: RNGKey) -> EnvParams:
        if self.frozen_prob is None:
            frozen = get_preset_map(self.shape)
            agent_start = jnp.array([[0, 0]])
            goal_start = jnp.array([[s - 1 for s in self.shape]])
        else:
            key, rng_p, rng_gen = jax.random.split(rng_key, 3)
            p_high = self.frozen_prob_high or self.frozen_prob
            p = jax.random.uniform(rng_p, minval=self.frozen_prob, maxval=p_high)
            frozen = generate_frozen_chunk(rng_gen, self.shape, p)
            agent_start = jnp.indices(self.shape)[:, frozen].T
            goal_start = jnp.indices(self.shape)[:, frozen].T
        return EnvParams(frozen, agent_start, goal_start)

    @partial(jax.jit, static_argnums=0)
    def reset(self, params: EnvParams, rng_key: RNGKey) -> tuple[EnvState, ObsType]:
        rng_key, rng_agent, rng_goal, rng_obs = jax.random.split(rng_key, 4)
        agent_pos = jax.random.choice(rng_agent, params.agent_start)
        goal_pos = jax.random.choice(rng_goal, params.goal_start)
        state = jax.lax.stop_gradient(EnvState(agent_pos, goal_pos))
        obs = self.get_obs(params, rng_obs, state)
        return state, obs

    @partial(jax.jit, static_argnums=0)
    def step(
        self, params: EnvParams, rng_key: RNGKey, state: EnvState, action: ActType
    ) -> tuple[EnvState, ObsType, float, bool, dict]:
        LEFT, DOWN, RIGHT, UP = jnp.array(0), jnp.array(1), jnp.array(2), jnp.array(3)
        delta = jnp.select(
            [action == LEFT, action == DOWN, action == RIGHT, action == UP],
            [jnp.array([0, -1]), jnp.array([1, 0]), jnp.array([0, 1]), jnp.array([-1, 0])],
        )
        new_agent_pos = jnp.clip(state.agent_pos + delta, 0, jnp.array(params.frozen.shape) - 1)
        state = EnvState(agent_pos=new_agent_pos, goal_pos=state.goal_pos)
        obs = self.get_obs(params, rng_key, state)

        reward, done = jax.lax.cond(
            (state.agent_pos == state.goal_pos).all(),
            lambda: (1.0, True),
            lambda: (0.0, ~params.frozen[tuple(state.agent_pos)]),
        )
        return state, obs, reward, done, {}

    @partial(jax.jit, static_argnums=0)
    def get_obs(self, params: EnvParams, rng_key: RNGKey, state: EnvState) -> ObsType:
        # one-hot encoding of the observation
        obs = jnp.zeros((*params.frozen.shape, 3))
        obs = obs.at[state.agent_pos[0], state.agent_pos[1], 0].set(1)
        obs = obs.at[state.goal_pos[0], state.goal_pos[1], 1].set(1)
        obs = obs.at[:, :, 2].set(~params.frozen)
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
