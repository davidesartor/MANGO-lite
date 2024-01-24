from __future__ import annotations
from functools import partial, wraps
from typing import Optional
import jax
import jax.numpy as jnp
from flax import struct

from policies import EpsilonGreedyPolicy, PolicyParams
from frozen_lake import EnvState, EnvParams, FrozenLake, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    done: bool
    info: dict


def get_rollout_fn(env: FrozenLake, policy: EpsilonGreedyPolicy):
    def rollout(env_params: EnvParams, policy_params: PolicyParams, rng_key: RNGKey, n_steps: int):
        def scan_compatible_step(carry, rng_key: RNGKey):
            env_state, obs = carry
            rng_key, rng_obs, rng_action, rng_step, rng_reset = jax.random.split(rng_key, 5)
            action: jax.Array = policy.apply(policy_params, rng_action, obs, method="get_action")
            next_env_state, next_obs, reward, done, info = env.step(
                env_params, rng_step, env_state, action
            )
            transition = Transition(env_state, obs, action, next_obs, reward, done, info)

            # reset the environment if done
            carry = jax.lax.cond(
                done,
                lambda: env.reset(env_params, rng_reset),
                lambda: (next_env_state, next_obs),
            )
            return carry, transition

        rng_key, rng_reset, rng_scan = jax.random.split(rng_key, 3)
        env_state, obs = env.reset(env_params, rng_reset)
        rng_steps = jax.random.split(rng_scan, n_steps)
        final_state, transitions = jax.lax.scan(scan_compatible_step, (env_state, obs), rng_steps)
        return transitions

    return wraps(rollout)(jax.jit(rollout, static_argnames=("n_steps",)))
