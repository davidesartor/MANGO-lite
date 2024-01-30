from functools import partial
from typing import Callable, Sequence
from flax import struct
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from frozen_lake import EnvState, FrozenLake, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    next_obs: ObsType
    reward: float
    done: bool
    info: dict


class RolloutManager(struct.PyTreeNode):
    env: FrozenLake
    qnet_apply_fn: Callable = struct.field(pytree_node=False)

    @partial(jax.jit, static_argnames=("n_steps"))
    def eps_greedy(self, params: optax.Params, rng_key: RNGKey, epsilon: float, n_steps: int):
        def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
            rng_eps, rng_action = jax.random.split(rng_key)
            qval = self.qnet_apply_fn(params, obs)
            return jax.lax.select(
                jax.random.uniform(rng_eps) < epsilon,
                jax.random.randint(rng_action, shape=(), minval=0, maxval=qval.size),
                qval.argmax(),
            )

        def scan_compatible_step(carry, rng_key: RNGKey):
            env_state, obs = carry
            rng_action, rng_step, rng_reset = jax.random.split(rng_key, 3)
            action = get_action(rng_action, obs)
            next_env_state, next_obs, reward, done, info = self.env.step(
                rng_step, env_state, action
            )
            transition = Transition(env_state, obs, action, next_obs, reward, done, info)

            # reset the environment if done
            carry = jax.lax.cond(
                done,
                lambda: self.env.reset(rng_reset),
                lambda: (next_env_state, next_obs),
            )
            return carry, transition

        rng_reset, *rng_steps = jax.random.split(rng_key, n_steps + 1)
        final_state, transitions = jax.lax.scan(
            scan_compatible_step, self.env.reset(rng_reset), jnp.array(rng_steps)
        )
        return transitions


class ConvNet(nn.Module):
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x):
        for ch in self.hidden:
            x = nn.Conv(ch, (3, 3))(x)
            x = nn.celu(x)
            x = nn.Conv(ch, (2, 2), strides=(2, 2))(x)
            x = nn.LayerNorm()(x)
        x = x.flatten()
        x = nn.Dense(features=self.out)(x)
        return x
