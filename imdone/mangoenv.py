from functools import partial
from flax import struct
import jax
import jax.numpy as jnp

import spaces
from utils import RNGKey, ObsType, ActType, FrozenLake, EnvState, Transition
from multiqlearning import MultiDQLTrainState


class MangoState(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType


class MangoEnv(struct.PyTreeNode):
    lower_layer: FrozenLake
    dql_state: MultiDQLTrainState
    action_space: spaces.Discrete = struct.field(pytree_node=False, default=spaces.Discrete(5))

    @jax.jit
    def reset(self, rng_key: RNGKey) -> tuple[MangoState, ObsType]:
        state, obs = self.lower_layer.reset(rng_key)
        return MangoState(state, obs), obs

    @partial(jax.jit, donate_argnames=("state",))
    def step(
        self, state: MangoState, rng_key: RNGKey, comand: ActType
    ) -> tuple[MangoState, ObsType, jnp.float32, jnp.bool_, dict]:
        def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
            qval = self.dql_state.qval_apply_fn(self.dql_state.params_qnet, obs)
            return qval[comand].argmax()

        def while_cond(carry):
            rng_key, state, beta, done, cum_reward = carry
            return ~(beta | done)

        def while_body(carry):
            rng_key, state, beta, done, cum_reward = carry
            rng_key, rng_action, rng_lower = jax.random.split(rng_key, 3)
            action = get_action(rng_action, state.obs)
            next_env_state, next_obs, reward, done, info = self.lower_layer.step(
                state.env_state, rng_lower, action
            )
            transition = Transition(state, state.obs, action, reward, next_obs, done, info)
            beta = self.dql_state.beta_fn(transition)
            state = MangoState(next_env_state, next_obs)
            return (rng_key, state, beta, done, cum_reward + reward)

        (rng_key, state, beta, done, cum_reward) = jax.lax.while_loop(
            while_cond, while_body, (rng_key, state, False, False, 0.0)
        )
        return state, state.obs, cum_reward, done, {}
