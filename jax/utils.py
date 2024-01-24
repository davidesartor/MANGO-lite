from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
from flax import struct

from policies import PolicyParams
from frozen_lake import EnvState, EnvParams, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    done: bool
    info: dict


def rollout(
    env_params: EnvParams,
    policy_params: PolicyParams,
    rng_key: RNGKey,
    n_steps: int,
    env_reset_fn: Callable[[EnvParams, RNGKey], EnvState],
    env_step_fn: Callable[
        [EnvParams, RNGKey, EnvState, ActType],
        tuple[EnvState, float, bool, dict],
    ],
    get_obs_fn: Callable[[EnvParams, RNGKey, EnvState], ObsType],
    get_action_fn: Callable[[PolicyParams, RNGKey, ObsType], ActType],
):
    def scan_compatible_step(carry, rng_key: RNGKey):
        env_state, obs = carry
        rng_key, rng_obs, rng_action, rng_step, rng_reset = jax.random.split(rng_key, 5)
        action = get_action_fn(policy_params, rng_action, obs)
        next_env_state, reward, done, info = env_step_fn(env_params, rng_step, env_state, action)
        next_obs = get_obs_fn(env_params, rng_obs, next_env_state)
        transition = Transition(env_state, obs, action, next_obs, reward, done, info)

        # reset the environment if done
        next_env_state = jax.lax.cond(
            done, lambda: env_reset_fn(env_params, rng_reset), lambda: next_env_state
        )
        next_obs = jax.lax.cond(
            done, lambda: get_obs_fn(env_params, rng_obs, next_env_state), lambda: next_obs
        )
        return (next_env_state, next_obs), transition

    rng_key, rng_reset, rng_scan = jax.random.split(rng_key, 3)
    env_state = env_reset_fn(env_params, rng_reset)
    obs = get_obs_fn(env_params, rng_key, env_state)
    rng_steps = jax.random.split(rng_scan, n_steps)
    final_state, transitions = jax.lax.scan(scan_compatible_step, (env_state, obs), rng_steps)
    return transitions
