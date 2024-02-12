from functools import partial
import spaces

import jax
import jax.numpy as jnp
from flax import struct

import maps


RNGKey = jax.Array
ObsType = jax.Array
ActType = jnp.int64


class EnvState(struct.PyTreeNode):
    agent_pos: jax.Array
    goal_pos: jax.Array


class FrozenLake(struct.PyTreeNode):
    frozen: jax.Array
    agent_start_prob: jax.Array
    goal_start_prob: jax.Array
    action_space: spaces.Discrete = struct.field(pytree_node=False, default=spaces.Discrete(4))

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "shape"))
    def make_preset(cls, rng_key: RNGKey, shape: tuple[int, int]):
        frozen = maps.get_preset_map(shape)
        agent_start_prob = jnp.zeros(frozen.shape).at[0, 0].set(1.0).flatten()
        goal_start_prob = jnp.zeros(frozen.shape).at[-1, -1].set(1.0).flatten()
        return cls(frozen, agent_start_prob, goal_start_prob)

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "shape"))
    def make_random(cls, rng_key: RNGKey, shape: tuple[int, int], frozen_prob: float):
        frozen = maps.generate_map(rng_key, shape, frozen_prob)
        agent_start_prob = (frozen.T / frozen.sum()).flatten()
        goal_start_prob = (frozen.T / frozen.sum()).flatten()
        return cls(frozen, agent_start_prob, goal_start_prob)

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "shape"))
    def make_mango_sandbox(cls, rng_key: RNGKey, shape: tuple[int, int], frozen_prob: float):
        rng_center, rng_edge = jax.random.split(rng_key)
        frozen = maps.generate_map(rng_center, shape, frozen_prob)
        frozen = jnp.pad(frozen, 1)
        agent_start_prob = (frozen.T / frozen.sum()).flatten()
        goal_start_prob = (frozen.T / frozen.sum()).flatten()
        edge = jax.random.uniform(rng_edge, frozen.shape) < frozen_prob
        edge = edge.at[1:-1, 1:-1].set(False)
        frozen = frozen | edge
        return cls(frozen, agent_start_prob, goal_start_prob)

    @jax.jit
    def reset(self, rng_key: RNGKey) -> tuple[EnvState, ObsType]:
        rng_agent, rng_goal, rng_obs = jax.random.split(rng_key, 3)
        agent_pos = jax.random.choice(
            rng_agent,
            jnp.indices(self.frozen.shape).T.reshape(-1, 2),
            p=self.agent_start_prob,
        )
        goal_pos = jax.random.choice(
            rng_goal,
            jnp.indices(self.frozen.shape).T.reshape(-1, 2),
            p=self.goal_start_prob,
        )
        state = jax.lax.stop_gradient(EnvState(agent_pos, goal_pos))
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
        # one-hot encoding of the observation
        obs = jnp.zeros((*self.frozen.shape, 3))
        obs = obs.at[state.agent_pos[0], state.agent_pos[1], 0].set(1)
        obs = obs.at[state.goal_pos[0], state.goal_pos[1], 1].set(1)
        obs = obs.at[:, :, 2].set(~self.frozen)
        return jax.lax.stop_gradient(obs)
