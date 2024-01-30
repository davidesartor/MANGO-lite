from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
import optax

import utils


class DDQNTrainState(TrainState):
    params_targ: optax.Params = struct.field(pytree_node=True)
    td_discount: float = struct.field(pytree_node=False, default=0.95)
    soft_update_rate: float = struct.field(pytree_node=False, default=0.005)

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable,
        params: optax.Params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)),
        **kwargs,
    ):
        params_targ = jax.tree_util.tree_map(lambda x: x, params)
        return super().create(
            apply_fn=apply_fn, params=params, params_targ=params_targ, tx=tx, **kwargs
        )

    @jax.jit
    def temporal_difference(
        self,
        params: optax.Params,
        params_targ: optax.Params,
        transition: utils.Transition,
    ) -> jax.Array:
        qstart = self.apply_fn(params, transition.obs)
        qselected = jnp.take(qstart, transition.action)
        qnext = self.apply_fn(params_targ, transition.next_obs)
        qnext = jax.lax.select(transition.done, 0.0, qnext.max())
        td = qselected - (transition.reward + self.td_discount * qnext)
        return td

    @partial(jax.jit, donate_argnames=("self",))
    def update(self, transitions: utils.Transition):
        def soft_update(params_targ, tau, params):
            return jax.tree_map(lambda pt, p: pt * (1 - tau) + p * tau, params_targ, params)

        def td_loss_fn(params, params_targ, transitions):
            batched_td_fn = jax.vmap(self.temporal_difference, in_axes=(None, None, 0))
            td = batched_td_fn(params, params_targ, transitions)
            return optax.squared_error(td).mean()

        td_gradients = jax.grad(td_loss_fn)(self.params, self.params_targ, transitions)
        new_params_targ = soft_update(self.params_targ, self.soft_update_rate, self.params)
        return self.apply_gradients(grads=td_gradients, params_targ=new_params_targ)
