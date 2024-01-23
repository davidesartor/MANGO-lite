from __future__ import annotations
from typing import Any, Callable, Optional
import jax
import jax.numpy as jnp
from flax import struct

from frozen_lake import EnvState, EnvParams, ObsType, ActType, RNGKey

PolicyParams = Any


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    reward: float
    done: bool
    info: dict


class SimulationState(struct.PyTreeNode):
    env_state: EnvState
    env_params: EnvParams = struct.field(pytree_node=False)
    policy_params: PolicyParams = struct.field(pytree_node=False)

    env_reset_fn: Callable[
        [RNGKey, EnvParams],
        EnvState,
    ] = struct.field(pytree_node=False)

    env_obs_fn: Callable[
        [RNGKey, EnvState, EnvParams],
        ObsType,
    ] = struct.field(pytree_node=False)

    env_step_fn: Callable[
        [RNGKey, EnvState, ActType, EnvParams],
        tuple[EnvState, float, bool, dict],
    ] = struct.field(pytree_node=False)

    policy_action_fn: Callable[
        [PolicyParams, RNGKey, ObsType],
        ActType,
    ] = struct.field(pytree_node=False)

    def step(self: SimulationState, rng_key: RNGKey):
        rng_key, rng_obs, rng_action, rng_step, rng_reset = jax.random.split(rng_key, 5)
        obs = self.env_obs_fn(rng_obs, self.env_state, self.env_params)
        action = self.policy_action_fn(self.policy_params, rng_action, obs)
        env_state, reward, done, info = self.env_step_fn(
            rng_step, self.env_state, action, self.env_params
        )
        transition = Transition(self.env_state, obs, action, reward, done, info)
        env_state = jax.lax.cond(
            done, lambda: self.env_reset_fn(rng_reset, self.env_params), lambda: env_state
        )
        return self.replace(env_state=env_state), transition

    def rollout(self: SimulationState, rng_key, steps):
        rng_steps = jax.random.split(rng_key, steps)
        final_state, transitions = jax.lax.scan(SimulationState.step, self, rng_steps)
        return final_state, transitions

    @classmethod
    def create(
        cls,
        rng_key,
        env,
        policy,
        env_params: Optional[EnvParams] = None,
        policy_params: Optional[PolicyParams] = None,
    ):
        rng_key, rng_env_init, rng_policy_init, rng_obs, rng_reset = jax.random.split(rng_key, 5)
        env_params = env_params or env.init(rng_env_init)
        env_state = env.reset(rng_reset, env_params)
        policy_params = policy_params or policy.init(
            rng_policy_init, rng_obs, env.obs(rng_obs, env_state, env_params)
        )
        return cls(
            env_params=env_params,
            policy_params=policy_params,
            env_state=env_state,
            env_reset_fn=env.reset,
            env_obs_fn=env.obs,
            env_step_fn=env.step,
            policy_action_fn=policy.apply,
        )
