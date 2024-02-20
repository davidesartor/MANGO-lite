from functools import partial
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

from utils import FrozenLake, EnvState, ObsType, ActType, RNGKey, Transition
import qlearning
import utils


class MultiDQLTrainState(qlearning.DQLTrainState):
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


@partial(jax.jit, donate_argnames=("sim_state",), static_argnames=("rollout_length", "train_iter"))
def multi_q_learning_step(sim_state, rng_key, rollout_length: int, train_iter: int):
    (env, dql_state, replay_memory) = sim_state
    rng_expl, rng_train, rng_eval = jax.random.split(rng_key, 3)

    exploration = utils.random_rollout(env, rng_expl, rollout_length)

    intrinsic_reward = jax.vmap(dql_state.reward_fn)(exploration)
    intrinsic_done = jax.vmap(dql_state.beta_fn)(exploration)
    exploration_repl = exploration.replace(reward=intrinsic_reward, done=intrinsic_done)

    replay_memory = replay_memory.push(exploration_repl)

    for rng_sample in jax.random.split(rng_train, train_iter):
        transitions = replay_memory.sample(rng_sample, min((rollout_length, 256)))
        dql_state = dql_state.update_params(transitions)

    # # evaluation rollout
    # multi_task_greedy_rollout = jax.vmap(
    #     greedy_rollout, in_axes=(None, None, None, None, 0), out_axes=0
    # )
    # evaluation = multi_task_greedy_rollout(env, dql_state, rng_eval, rollout_length, jnp.arange(5))

    # # use intrinsic signals
    # intrinsic_reward = jax.vmap(jax.vmap(dql_state.reward_fn), out_axes=-1)(evaluation)
    # intrinsic_done = jax.vmap(jax.vmap(dql_state.beta_fn), out_axes=-1)(evaluation)
    # evaluation = evaluation.replace(reward=intrinsic_reward, done=intrinsic_done)

    # # log results
    # results = SimResults(evaluation.reward, evaluation.done, exploration.reward, exploration.done)
    results = None
    return (env, dql_state, replay_memory), results
