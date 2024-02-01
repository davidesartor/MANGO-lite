from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

import utils
from utils import RNGKey, FrozenLake, EnvState, ObsType, ActType, Transition


class DQLTrainState(struct.PyTreeNode):
    params_qnet: optax.Params
    params_qnet_targ: optax.Params
    opt_state: optax.OptState

    qval_apply_fn: Callable = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    td_discount: float = struct.field(pytree_node=False, default=0.95)
    soft_update_rate: float = struct.field(pytree_node=False, default=0.005)
    step: int = 0

    @classmethod
    def create(cls, rng_key: RNGKey, qnet: nn.Module, obs: ObsType, optimizer=optax.adam(1e-3)):
        rng_qnet_init, rng_qnet_targ_init = jax.random.split(rng_key)
        params_qnet = qnet.init(rng_qnet_init, obs)
        params_qnet_targ = qnet.init(rng_qnet_targ_init, obs)
        opt_state = optimizer.init(params_qnet)
        return cls(params_qnet, params_qnet_targ, opt_state, qnet.apply, optimizer)

    @jax.jit
    def temporal_difference(
        self,
        params_qnet: optax.Params,
        params_qnet_targ: optax.Params,
        transition: Transition,
    ) -> jax.Array:
        qstart = self.qval_apply_fn(params_qnet, transition.obs)
        qselected = jnp.take(qstart, transition.action)
        qnext = self.qval_apply_fn(params_qnet_targ, transition.next_obs)
        qnext = jax.lax.select(transition.done, 0.0, qnext.max())
        td = qselected - (transition.reward + self.td_discount * qnext)
        return td

    @partial(jax.jit, donate_argnames=("self",))
    def update_params_qnet(self, transitions: Transition):
        def td_loss_fn(params_qnet, transitions):
            batched_td_fn = jax.vmap(self.temporal_difference, in_axes=(None, None, 0))
            td = batched_td_fn(params_qnet, self.params_qnet_targ, transitions)
            return optax.squared_error(td).mean()

        td_gradients = jax.grad(td_loss_fn)(self.params_qnet, transitions)
        updates, new_opt_state = self.optimizer.update(td_gradients, self.opt_state)
        new_params_qnet = optax.apply_updates(self.params_qnet, updates)

        def soft_update(params_qnet_targ, params_qnet, tau):
            return jax.tree_map(
                lambda pt, p: pt * (1 - tau) + p * tau, params_qnet_targ, params_qnet
            )

        new_params_qnet_targ = soft_update(
            self.params_qnet_targ, new_params_qnet, self.soft_update_rate
        )

        return self.replace(
            step=self.step + 1,
            params_qnet=new_params_qnet,
            params_qnet_targ=new_params_qnet_targ,
            opt_state=new_opt_state,
        )


@partial(jax.jit, static_argnames=("steps"))
def eps_greedy_rollout(
    env: FrozenLake, dql_state: DQLTrainState, rng_key: RNGKey, epsilon: float, steps: int
):
    def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
        qval = dql_state.qval_apply_fn(dql_state.params_qnet, obs)
        action = utils.eps_argmax(rng_key, qval, epsilon)
        return action

    def scan_compatible_step(carry, rng_key: RNGKey):
        env_state, obs = carry
        rng_action, rng_step, rng_reset = jax.random.split(rng_key, 3)
        action = get_action(rng_action, obs)
        next_env_state, next_obs, reward, done, info = env.step(env_state, rng_step, action)
        transition = Transition(env_state, obs, action, next_obs, reward, done, info)

        # reset the environment if done
        next_env_state, next_obs = jax.lax.cond(
            done,
            lambda: env.reset(rng_reset),
            lambda: (next_env_state, next_obs),
        )
        return (next_env_state, next_obs), transition

    rng_env_reset, rng_steps = jax.random.split(rng_key)
    rng_steps = jax.random.split(rng_steps, steps)
    env_state, obs = env.reset(rng_env_reset)
    _, transitions = jax.lax.scan(scan_compatible_step, (env_state, obs), rng_steps)
    return transitions
