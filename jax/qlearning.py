from typing import Any
from typing_extensions import Self
import jax
import jax.numpy as jnp

from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
import optax

import utils


class DDQNTrainState(TrainState):
    params_targ: FrozenDict[str, Any] = struct.field(pytree_node=True)
    td_discount: float = struct.field(pytree_node=False, default=0.95)
    soft_update_rate: float = struct.field(pytree_node=False, default=0.001)

    @classmethod
    def create(cls, *, params, tx=optax.adam(1e-3), **kwargs):
        return super().create(params=params, params_targ=params, tx=tx, **kwargs)

    @jax.jit
    def temporal_difference(
        self,
        params: FrozenDict[str, Any],
        params_targ: FrozenDict[str, Any],
        transition: utils.Transition,
    ) -> jax.Array:
        qstart = self.apply_fn(params, transition.obs)
        qselected = jnp.take(qstart, transition.action)
        qnext = self.apply_fn(params_targ, transition.next_obs)
        qnext = jax.lax.select(transition.done, 0.0, qnext.max())
        td = qselected - (transition.reward + self.td_discount * qnext)
        return td

    @jax.jit
    def td_gradient(self, transitions: utils.Transition) -> FrozenDict[str, Any]:
        def td_loss_fn(params, params_targ, transition):
            rollout_td_fn = jax.vmap(self.temporal_difference, in_axes=(None, None, 0))
            td = rollout_td_fn(params, params_targ, transition)
            return optax.huber_loss(td).mean()

        return jax.grad(td_loss_fn)(self.params, self.params_targ, transitions)

    @jax.jit
    def update_nets(self, grads: FrozenDict[str, Any]) -> Self:
        def soft_update(params_targ, tau, params):
            return jax.tree_map(lambda pt, p: pt * (1 - tau) + p * tau, params_targ, params)

        new_params_targ = soft_update(self.params_targ, self.soft_update_rate, self.params)
        return self.apply_gradients(grads=grads, params_targ=new_params_targ)
