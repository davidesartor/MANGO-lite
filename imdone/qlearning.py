from functools import partial
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

from utils import FrozenLake, EnvState, ObsType, ActType, RNGKey, Transition
import utils


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
    def create(cls, rng_key: RNGKey, qnet: nn.Module, sample_obs: ObsType, lr: float, **kwargs):
        rng_qnet, rng_qnet_targ = jax.random.split(rng_key)
        params_qnet = qnet.init(rng_qnet, sample_obs)
        params_qnet_targ = qnet.init(rng_qnet_targ, sample_obs)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params_qnet)
        return cls(params_qnet, params_qnet_targ, opt_state, qnet.apply, optimizer, **kwargs)

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
    def greedy_rollout(self, env: FrozenLake, rng_key: RNGKey, steps: int):
        def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
            qval = self.qval_apply_fn(self.params_qnet, obs)
            return qval.argmax()

        return utils.rollout(get_action, env, rng_key, steps)


class SimResults(NamedTuple):
    eval_reward: jax.Array
    eval_done: jax.Array
    expl_reward: jax.Array
    expl_done: jax.Array


@partial(jax.jit, donate_argnames=("sim_state",), static_argnames=("rollout_length", "train_iter"))
def q_learning_step(sim_state, rng_key, rollout_length: int, train_iter: int):
    (env, dql_state, replay_memory) = sim_state
    rng_expl, rng_train, rng_eval = jax.random.split(rng_key, 3)

    # exploration rollout
    exploration = utils.random_rollout(env, rng_expl, rollout_length)
    replay_memory = replay_memory.push(exploration)

    # # policy training
    for rng_sample in jax.random.split(rng_train, train_iter):
        transitions = replay_memory.sample(rng_sample, min((rollout_length, 256)))
        dql_state = dql_state.update_params(transitions)

    # evaluation rollout
    evaluation = dql_state.greedy_rollout(env, rng_eval, rollout_length // 2)

    # log results
    results = SimResults(evaluation.reward, evaluation.done, exploration.reward, exploration.done)
    return (env, dql_state, replay_memory), results
