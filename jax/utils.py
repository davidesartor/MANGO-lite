from __future__ import annotations
from functools import partial, wraps
from typing import Callable, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

from frozen_lake import EnvState, EnvParams, FrozenLake, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    done: bool
    info: dict


class ConvNet(nn.Module):
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x):
        a = self.param("a", nn.initializers.ones, (1, 1))
        for ch in self.hidden:
            x = nn.Conv(ch, (3, 3))(x)
            x = nn.celu(x)
            x = nn.Conv(ch, (2, 2), strides=(2, 2))(x)
            x = nn.LayerNorm()(x)
        x = x.flatten()
        x = nn.Dense(features=self.out)(x)
        return x


def eps_greedy_policy(qnet_apply_fn: Callable[[optax.Params, ObsType], jax.Array]):
    def get_action(params: optax.Params, epsilon: float, rng_key: RNGKey, obs: ObsType) -> ActType:
        rng_eps, rng_action = jax.random.split(rng_key)
        qval = qnet_apply_fn(params, obs)
        return jax.lax.cond(
            jax.random.uniform(rng_eps) < epsilon,
            lambda: jax.random.randint(rng_action, shape=(), minval=0, maxval=qval.size),
            lambda: qval.argmax(),
        )

    return wraps(get_action)(jax.jit(get_action))


def get_rollout_fn(
    env: FrozenLake,
    policy_apply_fn: Callable[[optax.Params, float, RNGKey, ObsType], ActType],
):
    def rollout(
        env_params: EnvParams,
        policy_params: optax.Params,
        epsilon: float,
        rng_key: RNGKey,
        n_steps: int,
    ):
        def scan_compatible_step(carry, rng_key: RNGKey):
            env_state, obs = carry
            rng_key, rng_obs, rng_action, rng_step, rng_reset = jax.random.split(rng_key, 5)
            action: jax.Array = policy_apply_fn(policy_params, epsilon, rng_action, obs)
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
