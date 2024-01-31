from functools import partial
from typing import Callable, Sequence
from flax import struct
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

import qlearning
import replay
from frozen_lake import EnvState, FrozenLake, ObsType, ActType, RNGKey


class MangoLayer(struct.PyTreeNode):
    qnet_apply_fn: Callable = struct.field(pytree_node=False)

    @jax.jit
    def get_action(
        self, params: optax.Params, rng_key: RNGKey, obs: ObsType, epsilon: float
    ) -> ActType:
        rng_eps, rng_action = jax.random.split(rng_key)
        qval = self.qnet_apply_fn(params, obs)
        return jax.lax.select(
            jax.random.uniform(rng_eps) < epsilon,
            jax.random.randint(rng_action, shape=(), minval=0, maxval=qval.size),
            qval.argmax(),
        )

    @partial(jax.jit, static_argnames=("n_steps"))
    def rollout(
        self,
        params: optax.Params,
        rng_key: RNGKey,
        env: FrozenLake,
        n_steps: int,
        epsilon: Sequence[float],
    ):
        def scan_compatible_step(carry, rng_key: RNGKey):
            env_state, obs, mango_state = carry
            rng_action, rng_step, rng_reset = jax.random.split(rng_key, 3)
            mango_state, action = self.get_action(params, mango_state, rng_action, obs, epsilon)
            next_env_state, next_obs, reward, done, info = env.step(rng_step, env_state, action)
            transition = Transition(env_state, obs, action, next_obs, reward, done, info)

            # reset the environment if done
            carry = jax.lax.cond(
                done,
                lambda: env.reset(rng_reset),
                lambda: (next_env_state, next_obs),
            )
            return carry, transition

        rng_reset, *rng_steps = jax.random.split(rng_key, n_steps + 1)
        final_state, transitions = jax.lax.scan(
            scan_compatible_step, env.reset(rng_reset), jnp.array(rng_steps)
        )
        return transitions
