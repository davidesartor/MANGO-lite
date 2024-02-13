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


@partial(jax.jit, static_argnames=("steps",))
def greedy_rollout(
    env: FrozenLake, dql_state: MultiDQLTrainState, rng_key: RNGKey, steps: int, task_id: int
):
    def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
        qval = dql_state.qval_apply_fn(dql_state.params_qnet, obs)
        return qval[task_id].argmax()

    return utils.rollout(get_action, env, rng_key, steps)


class SimResults(NamedTuple):
    eval_reward: jax.Array
    eval_done: jax.Array
    expl_reward: jax.Array
    expl_done: jax.Array


@partial(jax.jit, donate_argnames=("sim_state",), static_argnames=("rollout_length", "train_iter"))
def multi_q_learning_step(sim_state, rng_key, rollout_length: int, train_iter: int):
    (env, dql_state, replay_memory) = sim_state
    rng_expl, rng_train, rng_eval = jax.random.split(rng_key, 3)

    # exploration rollout
    exploration = utils.random_rollout(env, rng_expl, rollout_length)

    # use intrinsic signals
    intrinsic_reward = jax.vmap(dql_state.reward_fn)(exploration)
    intrinsic_done = jax.vmap(dql_state.beta_fn)(exploration)
    exploration = exploration.replace(reward=intrinsic_reward, done=intrinsic_done)

    # store exploration in replay memory
    replay_memory = replay_memory.push(exploration)

    # # policy training
    for rng_sample in jax.random.split(rng_train, train_iter):
        transitions = replay_memory.sample(rng_sample, min((rollout_length, 256)))
        dql_state = dql_state.update_params(transitions)

    # evaluation rollout
    multi_task_greedy_rollout = jax.vmap(greedy_rollout, in_axes=(None, None, None, None, -1))
    evaluation = multi_task_greedy_rollout(env, dql_state, rng_eval, rollout_length, jnp.arange(5))

    # use intrinsic signals
    intrinsic_reward = jax.vmap(dql_state.reward_fn)(evaluation)
    intrinsic_done = jax.vmap(dql_state.beta_fn)(evaluation)
    evaluation = evaluation.replace(reward=intrinsic_reward, done=intrinsic_done)

    # log results
    results = SimResults(evaluation.reward, evaluation.done, exploration.reward, exploration.done)
    return (env, dql_state, replay_memory), results
