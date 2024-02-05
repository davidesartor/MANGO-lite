from functools import partial
from typing import Callable, Sequence
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

import utils
from utils import RNGKey, EnvState, ObsType, ActType, Transition, FrozenLake
from qlearning import DQLTrainState


class MultiDQLTrainState(DQLTrainState):
    """DQLTrainState with multiple tasks.
    Transition reward should be a vector with one entry per task."""

    @jax.jit
    def temporal_difference(
        self,
        params_qnet: optax.Params,
        params_qnet_targ: optax.Params,
        transition: Transition,
    ) -> jax.Array:
        # transition rewards are a vector containing the reward for each policy
        qstart = self.qval_apply_fn(params_qnet, transition.obs)
        qselected = jax.vmap(jnp.take, in_axes=(0, None))(qstart, transition.action)
        qnext = self.qval_apply_fn(params_qnet_targ, transition.next_obs)
        qnext = jax.lax.select(transition.done, jnp.zeros_like(qnext), qnext).max(axis=-1)
        td = qselected - (transition.reward + self.td_discount * qnext)
        return td


class MangoDQLTrainState(struct.PyTreeNode):
    """DQLTrainState for MANGO architecture.
    outer_state: DQLTrainState for the agent's outer policy.
    inner_states: Sequence of MultiDQLTrainState for the inner layers.
    beta_fn: batched termination function for the inner policies.
    """

    outer: DQLTrainState
    inner: Sequence[MultiDQLTrainState]
    beta_fn: Callable[[Transition], jax.Array] = struct.field(pytree_node=False)
    reward_fn: Callable[[Transition], jax.Array] = struct.field(pytree_node=False)

    @partial(jax.jit, donate_argnames=("self",))
    def update_params_qnet(self, transitions: list[Transition]):
        outer = self.outer.update_params_qnet(transitions[0])
        inner = [state.update_params_qnet(t) for state, t in zip(self.inner, transitions[1:])]
        return self.replace(outer=outer, inner=inner)


@partial(jax.jit, static_argnames=("steps",))
def eps_greedy_rollout(
    env: FrozenLake,
    mango_dql_state: MangoDQLTrainState,
    rng_key: RNGKey,
    epsilons: jax.Array,
    steps: int,
):
    @partial(jax.jit, donate_argnames=("comands_under_exec",))
    def get_actions(rng_key, obs, comands_under_exec, betas, epsilons):
        """Traverse the policy DAG and return the path (actions at various layers)."""

        qvals_outer = mango_dql_state.outer.qval_apply_fn(mango_dql_state.outer.params_qnet, obs)
        qvals_inner = jnp.stack(
            [state.qval_apply_fn(state.params_qnet, obs) for state in mango_dql_state.inner]
        )

        rng_outer, rng_inner = jax.random.split(rng_key)
        rng_inner = jax.random.split(rng_inner, qvals_inner.shape[0])

        comand_want_execute = utils.eps_argmax(rng_outer, qvals_outer, epsilons[0])

        def scan_body(comand_want_execute, input):
            rng_key, qvals, epsilon, beta, comand_under_exec = input
            comand_executed = jax.lax.select(beta, comand_want_execute, comand_under_exec)
            action_want_execute = utils.eps_argmax(rng_key, qvals[comand_executed], epsilon)
            return action_want_execute, comand_executed

        scan_inputs = (rng_inner, qvals_inner, epsilons[1:], betas, comands_under_exec[:-1])
        action_executed, comands_executed = jax.lax.scan(
            scan_body, comand_want_execute, scan_inputs
        )
        return jnp.concatenate([comands_executed, action_executed[None,]])

    rng_env_reset, rng_scan = jax.random.split(rng_key)
    env_state, obs = env.reset(rng_env_reset)
    actions = jnp.zeros((1 + len(mango_dql_state.inner),), dtype=int)
    betas = jnp.ones((len(mango_dql_state.inner),), dtype=bool)

    def scan_body(carry, rng_key):
        env_state, obs, actions_prev, betas = carry

        rng_action, rng_step, rng_reset, rng_beta = jax.random.split(rng_key, 4)

        actions = get_actions(rng_action, obs, actions_prev, betas, epsilons)
        next_env_state, next_obs, reward, done, info = env.step(env_state, rng_step, actions[-1])

        transition_list = [
            Transition(env_state, obs, a, reward, next_obs, done, info) for a in actions
        ]
        betas = mango_dql_state.beta_fn(transition)

        transition_list = [
            Transition(env_state, obs, a, reward, next_obs, done, info) for a in actions
        ]

        # reset the environment if done
        next_env_state, next_obs, betas = jax.lax.cond(
            done,
            lambda: (*env.reset(rng_reset), jnp.ones_like(betas)),
            lambda: (next_env_state, next_obs, betas),
        )
        return (next_env_state, next_obs, actions, betas), (transition_list, betas)

    _, (transitions_lists, betas) = jax.lax.scan(
        scan_body, (env_state, obs, actions, betas), jax.random.split(rng_scan, steps)
    )
    return transitions_lists, betas


def aggregate(transitions, betas):
    def scan_body(future, input):
        current, beta = input
        beta = beta | current.done
        end_obs = jax.lax.select(beta, current.next_obs, future.next_obs)
        reward = jax.lax.select(beta, current.reward, current.reward + future.reward)
        aggr = current.replace(next_obs=end_obs, reward=reward, done=future.done | current.done)
        return aggr, aggr

    end = jax.tree_map(lambda x: x[-1], transitions)
    end = end.replace(reward=jnp.zeros_like(end.reward))
    _, aggr_transitions = jax.lax.scan(scan_body, end, (transitions, betas), reverse=True)
    return aggr_transitions
