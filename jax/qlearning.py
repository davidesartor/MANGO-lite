from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

from utils import epsilon_greedy_policy, FrozenLake, RNGKey, Transition


class DDQNTrainState(struct.PyTreeNode):
    params: optax.Params
    params_targ: optax.Params
    opt_state: optax.OptState

    qval_apply_fn: Callable = struct.field(pytree_node=False)
    action_apply_fn: Callable = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    env: FrozenLake = struct.field(pytree_node=False)

    td_discount: float = struct.field(pytree_node=False, default=0.95)
    soft_update_rate: float = struct.field(pytree_node=False, default=0.005)
    step: int = 0

    @classmethod
    def create(
        cls,
        rng_key: RNGKey,
        env: FrozenLake,
        qnet: nn.Module,
        optimizer=optax.adam(1e-3),
    ):
        rng_env_reset, rng_qnet_init, rng_qnet_targ_init = jax.random.split(rng_key, 3)
        env_state, env_obs = env.reset(rng_env_reset)
        params = qnet.init(rng_qnet_init, env_obs)
        params_targ = qnet.init(rng_qnet_targ_init, env_obs)
        opt_state = optimizer.init(params)
        action_apply_fn = epsilon_greedy_policy(qnet.apply)
        return cls(params, params_targ, opt_state, qnet.apply, action_apply_fn, optimizer, env)

    @jax.jit
    def temporal_difference(
        self,
        params: optax.Params,
        params_targ: optax.Params,
        transition: Transition,
    ) -> jax.Array:
        qstart = self.qval_apply_fn(params, transition.obs)
        qselected = jnp.take(qstart, transition.action)
        qnext = self.qval_apply_fn(params_targ, transition.next_obs)
        qnext = jax.lax.select(transition.done, 0.0, qnext.max())
        td = qselected - (transition.reward + self.td_discount * qnext)
        return td

    @partial(jax.jit, donate_argnames=("self",))
    def update_params(self, transitions: Transition):
        def td_loss_fn(params, transitions):
            batched_td_fn = jax.vmap(self.temporal_difference, in_axes=(None, None, 0))
            td = batched_td_fn(params, self.params_targ, transitions)
            return optax.squared_error(td).mean()

        td_gradients = jax.grad(td_loss_fn)(self.params, transitions)
        updates, new_opt_state = self.optimizer.update(td_gradients, self.opt_state)
        new_params = optax.apply_updates(self.params, updates)

        def soft_update(params_targ, params, tau):
            return jax.tree_map(lambda pt, p: pt * (1 - tau) + p * tau, params_targ, params)

        new_params_targ = soft_update(self.params_targ, new_params, self.soft_update_rate)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            params_targ=new_params_targ,
            opt_state=new_opt_state,
        )

    @partial(jax.jit, static_argnames=("steps"))
    def rollout(self, rng_key: RNGKey, steps: int, randomness: float):
        def scan_compatible_step(carry, rng_key: RNGKey):
            env_state, obs = carry
            rng_action, rng_step, rng_reset = jax.random.split(rng_key, 3)
            action = self.action_apply_fn(self.params, rng_action, obs, randomness)
            next_env_state, next_obs, reward, done, info = self.env.step(
                env_state, rng_step, action
            )
            transition = Transition(env_state, obs, action, next_obs, reward, done, info)

            # reset the environment if done
            next_env_state, next_obs = jax.lax.cond(
                done,
                lambda: self.env.reset(rng_reset),
                lambda: (next_env_state, next_obs),
            )
            return (next_env_state, next_obs), transition

        rng_env_reset, *rng_steps = jax.random.split(rng_key, steps + 1)
        env_state, obs = self.env.reset(rng_env_reset)
        final_state, transitions = jax.lax.scan(
            scan_compatible_step, (env_state, obs), jnp.array(rng_steps)
        )
        return transitions
