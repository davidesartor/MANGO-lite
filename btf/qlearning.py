from functools import partial
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

from frozen_lake import Env, RNGKey, Transition
import utils


class DQLTrainState(struct.PyTreeNode):
    params_qnet: optax.Params
    params_qnet_targ: optax.Params
    replay_buffer: utils.CircularBuffer
    opt_state: optax.OptState

    qval_apply_fn: Callable = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    td_discount: float = struct.field(pytree_node=False, default=0.95)
    soft_update_rate: float = struct.field(pytree_node=False, default=0.005)
    step: int = 0

    @classmethod
    def create(
        cls,
        rng_key: RNGKey,
        qnet: nn.Module,
        env: Env,
        *,
        lr=3e-4,
        replay_capacity=2**20,
        **kwargs
    ):
        rng_roll, rng_qnet, rng_qnet_targ = jax.random.split(rng_key, 3)
        sample_trans = jax.tree_map(lambda x: x[0], utils.random_rollout(env, rng_roll, 2))

        params_qnet = qnet.init(rng_qnet, sample_trans.obs)
        params_qnet_targ = qnet.init(rng_qnet_targ, sample_trans.obs)

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params_qnet)

        replay_buffer = utils.CircularBuffer.create(sample_trans, capacity=replay_capacity)
        return cls(
            params_qnet, params_qnet_targ, replay_buffer, opt_state, qnet.apply, optimizer, **kwargs
        )

    @jax.jit
    def temporal_difference(
        self,
        params_qnet: optax.Params,
        params_qnet_targ: optax.Params,
        transition: Transition,
    ) -> jax.Array:
        qstart = self.qval_apply_fn(params_qnet, transition.obs)
        qselected = qstart[transition.action]
        qnext = self.qval_apply_fn(params_qnet_targ, transition.next_obs)
        qnext = jax.lax.select(transition.done, 0.0, qnext.max())
        td = qselected - (transition.reward + self.td_discount * qnext)
        return td

    def process_transitions(self, transitions: Transition) -> Transition:
        return transitions

    @partial(jax.jit, donate_argnames=("self",))
    def update_replay(self, transitions: Transition):
        return self.replace(replay_buffer=self.replay_buffer.extend(transitions))

    @partial(jax.jit, donate_argnames=("self",))
    def update_params(self, transitions: Transition):
        def td_loss_fn(params_qnet, transitions):
            batched_td_fn = jax.vmap(self.temporal_difference, in_axes=(None, None, 0))
            td = batched_td_fn(params_qnet, self.params_qnet_targ, transitions)
            return optax.squared_error(td).mean()

        td_gradients = jax.grad(td_loss_fn)(self.params_qnet, transitions)
        updates, new_opt_state = self.optimizer.update(td_gradients, self.opt_state)
        new_params_qnet = optax.apply_updates(self.params_qnet, updates)

        def soft_update(target, source, tau):
            return jax.tree_map(lambda pt, p: pt * (1 - tau) + p * tau, target, source)

        new_params_qnet_targ = soft_update(
            self.params_qnet_targ, new_params_qnet, self.soft_update_rate
        )

        return self.replace(
            step=self.step + 1,
            params_qnet=new_params_qnet,
            params_qnet_targ=new_params_qnet_targ,
            opt_state=new_opt_state,
        )

    @partial(jax.jit, static_argnames=("steps",))
    def greedy_rollout(self, env: Env, rng_key: RNGKey, steps: int):
        def greedy_action(rng_key, obs):
            qvals = self.qval_apply_fn(self.params_qnet, obs)
            return qvals.argmax()

        return utils.rollout(greedy_action, env, rng_key, steps)


class MultiDQLTrainState(DQLTrainState):
    reward_fn: Callable[[Transition], jax.Array] = struct.field(
        pytree_node=False, default=lambda t: t.reward
    )
    beta_fn: Callable[[Transition], bool] = struct.field(
        pytree_node=False, default=lambda t: t.done
    )

    @jax.jit
    def temporal_difference(
        self,
        params_qnet: optax.Params,
        params_qnet_targ: optax.Params,
        transition: Transition,
    ) -> jax.Array:
        qstart = self.qval_apply_fn(params_qnet, transition.obs)
        qselected = qstart[:, transition.action]
        qnext = self.qval_apply_fn(params_qnet_targ, transition.next_obs)
        qnext = jax.lax.select(transition.done, jnp.zeros_like(qselected), qnext.max(axis=-1))
        td = qselected - (transition.reward + self.td_discount * qnext)
        return td

    @jax.jit
    def process_transitions(self, transitions: Transition) -> Transition:
        return transitions.replace(
            reward=jax.vmap(self.reward_fn)(transitions),
            done=jax.vmap(self.beta_fn)(transitions),
        )

    @partial(jax.jit, static_argnames=("steps", "task_id"))
    def greedy_rollout(self, env: Env, rng_key: RNGKey, steps: int, task_id=-1):
        def greedy_action(rng_key, obs):
            qvals = self.qval_apply_fn(self.params_qnet, obs)
            return qvals[task_id].argmax()

        return utils.rollout(greedy_action, env, rng_key, steps)
