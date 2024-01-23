from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import struct

from policies import DQNPolicy, PolicyParams
from frozen_lake import EnvState, EnvParams, FrozenLake, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    reward: float
    done: bool
    info: dict


def rollout(
    env: FrozenLake,
    policy: DQNPolicy,
    rng_key: RNGKey,
    policy_params: PolicyParams,
    steps: int,
    env_params: EnvParams,
):
    def scan_compatible_step(env_state: EnvState, rng_key: RNGKey):
        rng_key, rng_obs, rng_action, rng_step, rng_reset = jax.random.split(rng_key, 5)
        obs = env.get_obs(rng_obs, env_state, env_params)
        action = policy.apply(policy_params, rng_action, obs)
        next_env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
        transition = Transition(env_state, obs, action, reward, done, info)
        next_env_state = jax.lax.cond(
            done,
            lambda: env.reset(rng_reset, env_params),
            lambda: next_env_state,
        )
        return next_env_state, transition

    rng_key, rng_reset, rng_scan = jax.random.split(rng_key, 3)
    env_state = env.reset(rng_reset, env_params)
    rng_steps = jax.random.split(rng_scan, steps)

    final_state, transitions = jax.lax.scan(scan_compatible_step, env_state, rng_steps)
    return transitions


def loss_fn(policy, policy_params, transitions, discount=0.95):
    def qval(transition):
        return policy.apply(policy_params, transition.obs, method="qval")

    qvals: jax.Array = jax.vmap(qval)(transitions)  # type: ignore
    qselected = jnp.take_along_axis(qvals[:-1], transitions.action[:-1, None], axis=-1).squeeze(-1)
    qnext = jnp.max(qvals[1:], axis=-1)
    td = qselected + transitions.reward[:-1] - discount * qnext * (1 - transitions.done[:-1])
    return jnp.mean(td**2)
