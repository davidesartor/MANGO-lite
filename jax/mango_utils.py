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


def soft_update(params_qnet_targ, params_qnet, tau):
    return jax.tree_map(lambda pt, p: pt * (1 - tau) + p * tau, params_qnet_targ, params_qnet)


class MangoTransition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    actions: jax.Array
    rewards: jax.Array
    next_obs: ObsType
    betas: jax.Array
    done: jax.Array


class MultiDQLTrainState(struct.PyTreeNode):
    """DQLTrainState with multiple tasks.
    Transition reward should be a vector with one entry per task."""

    params_qnet: optax.Params
    params_qnet_targ: optax.Params
    opt_state: optax.OptState

    qval_apply_fn: Callable = struct.field(pytree_node=False)
    beta_fn: Callable = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    td_discount: float = struct.field(pytree_node=False, default=0.95)
    soft_update_rate: float = struct.field(pytree_node=False, default=0.005)
    step: int = 0

    @classmethod
    def create(
        cls,
        rng_key: RNGKey,
        qnet: nn.Module,
        obs: ObsType,
        beta_fn: Callable,
        optimizer=optax.adam(1e-3),
    ):
        rng_qnet_init, rng_qnet_targ_init = jax.random.split(rng_key)
        params_qnet = qnet.init(rng_qnet_init, obs)
        params_qnet_targ = qnet.init(rng_qnet_targ_init, obs)
        opt_state = optimizer.init(params_qnet)
        return cls(params_qnet, params_qnet_targ, opt_state, qnet.apply, beta_fn, optimizer)

    @jax.jit
    def temporal_difference(
        self,
        params_qnet: optax.Params,
        params_qnet_targ: optax.Params,
        transition: MangoTransition,
    ) -> jax.Array:
        qstart = self.qval_apply_fn(params_qnet, transition.obs)
        qnext = self.qval_apply_fn(params_qnet_targ, transition.next_obs)

        def single_task_td(qstart, action, reward, qnext, beta):
            qnext = jax.lax.select(beta, 0.0, qnext.max())
            td = qstart[action] - (reward + self.td_discount * qnext)
            return td

        # tasks in the same layer share the same action and beta, so no map
        multi_task_td = jax.vmap(single_task_td, in_axes=(0, None, 0, 0, None))
        multi_layer_td = jax.vmap(multi_task_td)

        td = multi_layer_td(qstart, transition.actions, transition.rewards, qnext, transition.betas)
        return td

    @partial(jax.jit, donate_argnames=("self",))
    def update_params(self, transitions: MangoTransition):
        def td_loss_fn(params_qnet):
            batched_td_fn = jax.vmap(self.temporal_difference, in_axes=(None, None, 0))
            td = batched_td_fn(params_qnet, self.params_qnet_targ, transitions)
            return optax.squared_error(td).mean()

        td_gradients = jax.grad(td_loss_fn)(self.params_qnet)
        updates, new_opt_state = self.optimizer.update(td_gradients, self.opt_state)
        new_params_qnet = optax.apply_updates(self.params_qnet, updates)
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
def eps_greedy_rollout(
    env: FrozenLake,
    dql_state: MultiDQLTrainState,
    rng_key: RNGKey,
    epsilons: jax.Array,
    steps: int,
):
    @partial(jax.jit, donate_argnames=("actions_under_exec",))
    def get_actions(rng_key, obs, actions_under_exec, betas, epsilons, outer_comand=-1):
        """Traverse the policy DAG and return the path (actions at various layers)."""

        def scan_body(comand_executed, input):
            rng_key, qvals, epsilon, beta, action_under_exec = input
            action_want_execute = utils.eps_argmax(rng_key, qvals[comand_executed], epsilon)
            action_executed = jax.lax.select(beta, action_want_execute, action_under_exec)
            return action_executed, action_executed

        rng_layers = jax.random.split(rng_key, epsilons.size)
        qvals = dql_state.qval_apply_fn(dql_state.params_qnet, obs)  # (layers, tasks, actions)
        # comands are unfrozen based on the next layer stop condition
        betas = jnp.roll(betas, shift=-1).at[-1].set(True)  # last layer implicitly terminates
        scan_inputs = (rng_layers, qvals, epsilons, betas, actions_under_exec)
        _, new_actions = jax.lax.scan(scan_body, outer_comand, scan_inputs)
        return new_actions

    def scan_body(carry, rng_key):
        env_state, obs, actions_under_exec, betas = carry

        rng_action, rng_step, rng_reset = jax.random.split(rng_key, 3)
        obs_stacked = jnp.repeat(obs[jnp.newaxis], epsilons.size, axis=0)
        actions = get_actions(rng_action, obs_stacked, actions_under_exec, betas, epsilons)
        next_env_state, next_obs, reward, done, info = env.step(env_state, rng_step, actions[-1])
        transition = Transition(env_state, obs, actions, reward, next_obs, done, info)
        betas, rewards = dql_state.beta_fn(transition)
        # atomic actions are not trained and always terminate after execution
        betas = jax.lax.select(done, jnp.ones_like(betas), betas)
        next_obs_stacked = jnp.repeat(next_obs[jnp.newaxis], epsilons.size, axis=0)
        mango_transition = MangoTransition(
            env_state, obs_stacked, actions, rewards, next_obs_stacked, betas, done
        )

        # reset the environment if done
        next_env_state, next_obs = jax.lax.cond(
            done, lambda: env.reset(rng_reset), lambda: (next_env_state, next_obs)
        )
        return (next_env_state, next_obs, actions, betas), mango_transition

    rng_env_reset, rng_scan = jax.random.split(rng_key)
    env_state, obs = env.reset(rng_env_reset)
    actions = jnp.zeros(epsilons.size, dtype=int)
    betas = jnp.ones(epsilons.size, dtype=bool)

    _, mango_transitions = jax.lax.scan(
        scan_body, (env_state, obs, actions, betas), jax.random.split(rng_scan, steps)
    )
    mango_transitions = aggregate(mango_transitions)
    return mango_transitions


def aggregate(transitions):
    def scan_body(future, current):
        # at upper layers, accumulate all transitions in one lower step
        low_beta = current.betas[1:]
        end_obs = current.next_obs.at[:-1].set(
            jnp.where(low_beta, current.next_obs[:-1], future.next_obs[:-1])
        )
        rewards = current.rewards.at[:-1].set(
            jnp.where(low_beta, current.rewards[:-1], future.rewards[:-1] + current.rewards[:-1])
        )
        aggr = current.replace(next_obs=end_obs, rewards=rewards)
        return aggr, aggr

    end = jax.tree_map(lambda x: x[-1], transitions)
    end = end.replace(rewards=jnp.zeros_like(end.rewards), betas=jnp.zeros_like(end.betas))
    _, aggr_transitions = jax.lax.scan(scan_body, end, transitions, reverse=True)
    return aggr_transitions
