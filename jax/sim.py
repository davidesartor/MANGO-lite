from functools import partial
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
import optax
from tqdm.auto import tqdm

import frozen_lake
import utils
import replay
import qlearning
import actions
import mango_utils


class Results(NamedTuple):
    eval_reward: jax.Array
    eval_done: jax.Array
    expl_reward: jax.Array
    expl_done: jax.Array


def run_q_learning_simulation(
    rng_key: jax.Array,
    map_size: int,
    n_rollouts: int,
    rollout_length: int,
    train_iter: int,
    lr: float,
    annealing_fn: Callable[[int], float],
):
    def simulation_setup(rng_key: jax.Array, map_size):
        rng_env_init, rng_env_reset, rng_dql = jax.random.split(rng_key, 3)
        env = frozen_lake.FrozenLake.make(rng_env_init, (map_size, map_size))
        env_state, env_obs = env.reset(rng_env_reset)

        qnet = utils.ConvNet(hidden=[2 * map_size] * int(jnp.log2(map_size)), out=4)
        dql_state = qlearning.DQLTrainState.create(rng_dql, qnet, env_obs, optax.adam(lr))

        init_transitions = qlearning.eps_greedy_rollout(
            env, dql_state, rng_key, epsilon=0.0, steps=2
        )
        replay_memory = replay.CircularBuffer.create(
            init_transitions, capacity=train_iter * rollout_length
        )
        return env, dql_state, replay_memory

    rng_init, rng_loop = jax.random.split(rng_key)
    env, dql_state, replay_memory = simulation_setup(rng_init, map_size)

    def loop_body(carry, rng_loop_iter):
        dql_state, replay_memory = carry
        # bind progress bar update host callback to rng_key split
        rng_expl, rng_train, rng_eval = host_callback.id_tap(
            lambda a, t: pbar.update(1), jax.random.split(rng_loop_iter, 3)
        )

        # exploration rollout
        expl_transitions = qlearning.eps_greedy_rollout(
            env, dql_state, rng_expl, annealing_fn(dql_state.step), rollout_length
        )
        replay_memory = replay_memory.push(expl_transitions)

        # policy training
        for rng_sample in jax.random.split(rng_train, train_iter):
            train_transitions = replay_memory.sample(rng_sample, rollout_length)
            dql_state = dql_state.update_params(train_transitions)

        # evaluation rollout
        eval_transitions = qlearning.eps_greedy_rollout(
            env, dql_state, rng_eval, 0.0, rollout_length
        )

        # step results
        step_results = Results(
            eval_transitions.reward,
            eval_transitions.done,
            expl_transitions.reward,
            expl_transitions.done,
        )
        return (dql_state, replay_memory), step_results

    pbar = tqdm(total=n_rollouts)
    (dql_state, replay_memory), results = jax.lax.scan(
        f=loop_body,
        init=(dql_state, replay_memory),
        xs=jax.random.split(rng_loop, n_rollouts),
    )
    pbar.close()
    return env, dql_state, results


def run_mango_simulation(
    rng_key: jax.Array,
    map_size: int,
    n_rollouts: int,
    rollout_length: int,
    train_iter: int,
    lr: float,
    annealing_fn: Callable[[int], jax.Array],
):
    def simulation_setup(rng_key: jax.Array, map_size):
        rng_env_init, rng_env_reset, rng_dql = jax.random.split(rng_key, 3)
        env = frozen_lake.FrozenLake.make(rng_env_init, (map_size, map_size))
        env_state, env_obs = env.reset(rng_env_reset)

        n_actions = 5
        n_layers = int(jnp.log2(map_size))

        cells = jnp.array([(2**i, 2**i) for i in reversed(range(1, n_layers + 1))])
        dql_state = mango_utils.MultiDQLTrainState.create(
            rng_dql,
            utils.MultiLayerConvNet([2 * map_size] * n_layers, n_actions),
            obs=jnp.stack([env_obs] * n_layers),
            beta_fn=partial(actions.beta_fn, cells),
            optimizer=optax.adam(lr),
        )

        epsilons = annealing_fn(dql_state.step)
        init_transitions = mango_utils.eps_greedy_rollout(env, dql_state, rng_key, epsilons, 2)
        replay_memory = replay.CircularBuffer.create(
            init_transitions, capacity=train_iter * rollout_length
        )
        return env, dql_state, replay_memory

    rng_init, rng_loop = jax.random.split(rng_key)
    env, dql_state, replay_memory = simulation_setup(rng_init, map_size)

    def loop_body(carry, rng_loop_iter):
        dql_state, replay_memory = carry
        # bind progress bar update host callback to rng_key split
        rng_expl, rng_train, rng_eval = host_callback.id_tap(
            lambda a, t: pbar.update(1), jax.random.split(rng_loop_iter, 3)
        )

        # exploration rollout
        epsilons = annealing_fn(dql_state.step)
        expl_transitions = mango_utils.eps_greedy_rollout(
            env, dql_state, rng_expl, epsilons, rollout_length
        )
        aggr_expl_transitions = mango_utils.aggregate(expl_transitions)
        replay_memory = replay_memory.push(aggr_expl_transitions)

        # policy training
        for rng_sample in jax.random.split(rng_train, train_iter):
            train_transitions = replay_memory.sample(rng_sample, rollout_length)
            dql_state = dql_state.update_params(train_transitions)

        # evaluation rollout
        eval_transitions = mango_utils.eps_greedy_rollout(
            env, dql_state, rng_eval, jnp.zeros_like(epsilons), rollout_length
        )

        # step results
        step_results = Results(
            eval_transitions.rewards,
            eval_transitions.done,
            expl_transitions.rewards,
            expl_transitions.done,
        )
        return (dql_state, replay_memory), step_results

    pbar = tqdm(total=n_rollouts)
    (dql_state, replay_memory), results = jax.lax.scan(
        f=loop_body,
        init=(dql_state, replay_memory),
        xs=jax.random.split(rng_loop, n_rollouts),
    )
    pbar.close()
    return env, dql_state, results
