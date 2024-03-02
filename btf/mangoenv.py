from functools import partial
from flax import struct
import jax
import jax.numpy as jnp
from typing import Callable

import spaces
import qlearning
from frozen_lake import RNGKey, ObsType, ActType, Env, EnvState, Transition


class MangoEnv(struct.PyTreeNode):
    lower_layer: Env
    dql_state: qlearning.MultiDQLTrainState
    max_steps: jnp.int_ = struct.field(pytree_node=False, default=jnp.inf)
    action_space: spaces.Discrete = struct.field(pytree_node=False, default=spaces.Discrete(5))

    @jax.jit
    def reset(self, rng_key: RNGKey):
        return self.lower_layer.reset(rng_key)

    @jax.jit
    def get_obs(self, rng_key: RNGKey, env_state: EnvState) -> ObsType:
        return self.lower_layer.get_obs(rng_key, env_state)

    @jax.jit
    def get_action(self, comand, rng_key: RNGKey, obs: ObsType) -> ActType:
        qval = self.dql_state.qval_apply_fn(self.dql_state.params_qnet, obs)
        return qval[comand].argmax()

    @partial(jax.jit, donate_argnames=("env_state",))
    def step(
        self, env_state: EnvState, rng_key: RNGKey, comand: ActType
    ) -> tuple[EnvState, ObsType, jnp.float32, jnp.bool_, dict]:
        def while_cond(carry):
            i, rng_key, env_state, obs, cum_reward, done, beta = carry
            return ~(beta | done)

        def while_body(carry):
            i, rng_key, env_state, obs, cum_reward, done, beta = carry
            rng_key, rng_action, rng_lower = jax.random.split(rng_key, 3)

            action = self.get_action(comand, rng_action, obs)
            next_env_state, next_obs, reward, done, info = self.lower_layer.step(
                env_state, rng_lower, action
            )
            transition = Transition(env_state, obs, action, reward, next_obs, done, info)
            beta = self.dql_state.beta_fn(transition)
            done = done | (i > self.max_steps)
            return (i + 1, rng_key, next_env_state, next_obs, cum_reward + reward, done, beta)

        rng_key, rng_obs = jax.random.split(rng_key)
        obs = self.lower_layer.get_obs(rng_obs, env_state)
        init_val = (0, rng_key, env_state, obs, 0.0, False, False)
        (steps, _, env_state, obs, cum_reward, done, _) = jax.lax.while_loop(
            while_cond, while_body, init_val
        )
        return env_state, obs, cum_reward, done, {}
