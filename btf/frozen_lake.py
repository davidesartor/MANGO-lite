from functools import partial
from typing import Protocol
import spaces

import jax
import jax.numpy as jnp
from flax import struct

RNGKey = jax.Array
ObsType = jax.Array
ActType = jnp.int_


class EnvState(struct.PyTreeNode):
    agent_pos: jax.Array
    goal_pos: jax.Array


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    reward: float
    next_obs: ObsType
    done: bool
    info: dict


class Env(Protocol):
    @property
    def action_space(self) -> spaces.Discrete:
        ...

    def reset(self, rng_key: RNGKey) -> tuple[EnvState, ObsType]:
        ...

    def step(
        self, state: EnvState, rng_key: RNGKey, action: ActType
    ) -> tuple[EnvState, ObsType, float, bool, dict]:
        ...

    @jax.jit
    def get_obs(self, rng_key: RNGKey, state: EnvState) -> ObsType:
        ...


class FrozenLake(struct.PyTreeNode):
    frozen: jax.Array
    agent_start_prob: jax.Array
    goal_start_prob: jax.Array
    action_space: spaces.Discrete = struct.field(pytree_node=False, default=spaces.Discrete(4))

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "scale"))
    def make_preset(cls, rng_key: RNGKey, scale: int, random_start=False):
        frozen = get_preset_map(scale)
        agent_start_prob = jax.lax.select(
            random_start,
            (frozen / frozen.sum()).flatten(),
            jnp.zeros(frozen.size).at[0].set(1.0),
        )
        goal_start_prob = jnp.zeros(frozen.size).at[-1].set(1.0)
        return cls(frozen, agent_start_prob, goal_start_prob)

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "scale"))
    def make_random(cls, rng_key: RNGKey, scale, frozen_prob: float):
        frozen = generate_chunks(rng_key, scale, frozen_prob)[0]
        agent_start_prob = (frozen / frozen.sum()).flatten()
        goal_start_prob = (frozen / frozen.sum()).flatten()
        return cls(frozen, agent_start_prob, goal_start_prob)

    @jax.jit
    def reset(self, rng_key: RNGKey) -> tuple[EnvState, ObsType]:
        rng_agent, rng_goal, rng_obs = jax.random.split(rng_key, 3)
        agent_pos = jnp.unravel_index(
            jax.random.choice(rng_agent, self.frozen.size, p=self.agent_start_prob),
            shape=self.frozen.shape,
        )
        goal_pos = jnp.unravel_index(
            jax.random.choice(rng_goal, self.frozen.size, p=self.goal_start_prob),
            shape=self.frozen.shape,
        )
        state = EnvState(jnp.array(agent_pos), jnp.array(goal_pos))
        obs = self.get_obs(rng_obs, state)
        return state, obs

    @partial(jax.jit, donate_argnames=("state",))
    def step(
        self, state: EnvState, rng_key: RNGKey, action: ActType
    ) -> tuple[EnvState, ObsType, jnp.float32, jnp.bool_, dict]:
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])[action]
        new_agent_pos = jnp.clip(state.agent_pos + delta, 0, jnp.array(self.frozen.shape) - 1)
        state = state.replace(agent_pos=new_agent_pos)
        obs = self.get_obs(rng_key, state)
        reward = jax.lax.select((state.agent_pos == state.goal_pos).all(), 1.0, 0.0)
        done = (state.agent_pos == state.goal_pos).all() | ~self.frozen[tuple(state.agent_pos)]
        return state, obs, reward, done, {}

    @jax.jit
    def get_obs(self, rng_key: RNGKey, state: EnvState) -> ObsType:
        return jnp.stack([state.agent_pos, state.goal_pos], axis=0)
        # one-hot encoding of the observation
        obs = jnp.zeros((*self.frozen.shape, 2))
        obs = obs.at[state.agent_pos[0], state.agent_pos[1], 0].set(1)
        obs = obs.at[state.goal_pos[0], state.goal_pos[1], 1].set(1)
        # obs = obs.at[:, :, 2].set(~self.frozen)
        return obs


@jax.jit
def connections(chunks: jax.Array):
    """Take an RNGKey of shape (4, x, x)
    representing 4 cells in a 4x4 grid arranged as [[0, 1], [2, 3]],
    retuns a boolean array with the possible connections between the cells."""

    def has_passage(left_edge: jax.Array, right_edge: jax.Array):
        return (left_edge & right_edge).any().astype(bool)

    UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT = chunks
    connections = (
        has_passage(UPPER_LEFT[-1:, :], LOWER_LEFT[:1, :]),
        has_passage(LOWER_LEFT[:, -1:], LOWER_RIGHT[:, :1]),
        has_passage(UPPER_RIGHT[-1:, :], LOWER_RIGHT[:1, :]),
        has_passage(UPPER_LEFT[:, -1:], UPPER_RIGHT[:, :1]),
    )
    return jnp.array(connections)


@jax.jit
def unable_to_merge(chunks: jax.Array):
    """Take an RNGKey of shape (4, x, x)
    representing 4 cells in a 4x4 grid arranged as [[0, 1], [2, 3]],
    and checks if their combination results in a valid chunk."""

    if chunks.size == 4:
        diag1 = jnp.array([[1, 0, 0, 1]])
        diag2 = jnp.array([[0, 1, 1, 0]])
        has_no_holes = chunks.sum() == 0
        is_diag = (chunks == diag1).all() | (chunks == diag2).all()
        return has_no_holes | is_diag
    else:
        return connections(chunks).sum() < 3


@partial(jax.jit, static_argnames=("scale", "n"))
def generate_chunks(rng_key: RNGKey, scale: int, p: float, n: int = 1):
    """Generates a (2**scale, 2**scale) valid chunk."""

    if scale == 0:
        return jax.random.uniform(rng_key, (n,)) < p

    def generate_single(rng_key: RNGKey):
        def while_cond(carry):
            rng_key, subchunks = carry
            return unable_to_merge(subchunks)

        def while_body(carry):
            rng_key, subchunks = carry
            rng_key, subkey = jax.random.split(rng_key)
            subchunks = generate_chunks(subkey, scale - 1, p, 4)
            return rng_key, subchunks

        rng_key, subchunks = while_body((rng_key, None))
        _, subchunks = jax.lax.while_loop(while_cond, while_body, (rng_key, subchunks))
        UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT = subchunks
        map = jnp.block([[UPPER_LEFT, UPPER_RIGHT], [LOWER_LEFT, LOWER_RIGHT]])
        return map

    return jax.vmap(generate_single)(jax.random.split(rng_key, n))


@partial(jax.jit, static_argnames=("scale",))
def get_preset_map(scale: int) -> jax.Array:
    if scale == 1:
        map = ["FH", "FF"]
    elif scale == 2:
        map = ["FFFF", "FHFH", "FFFH", "HFFF"]
    elif scale == 3:
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
        raise ValueError(f"no preset map for scale {scale}")
    return jnp.array([[c == "F" for c in row] for row in map])
