from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
from tqdm.auto import tqdm

import frozen_lake
import utils
import replay
from qlearning import DQLTrainState, eps_greedy_rollout


def simulation_setup(rng_key: jax.Array, map_size):
    rng_env_init, rng_env_reset, rng_dql = jax.random.split(rng_key, 3)
    env = frozen_lake.FrozenLake.make(rng_env_init, (map_size, map_size))
    env_state, env_obs = env.reset(rng_env_reset)

    qnet = utils.ConvNet(hidden=[2 * map_size] * int(jnp.log2(map_size)), out=4)
    dql_state = DQLTrainState.create(rng_dql, qnet, env_obs)

    init_transitions = eps_greedy_rollout(env, dql_state, rng_key, epsilon=0.0, steps=2)
    replay_memory = replay.CircularBuffer.create(init_transitions)
    return env, dql_state, replay_memory


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
    eps_annealing: Callable[[int], float],
):
    rng_init, rng_loop = jax.random.split(rng_key)
    env, dql_state, replay_memory = simulation_setup(rng_init, map_size)

    def loop_body(carry, rng_loop_iter):
        dql_state, replay_memory = carry
        # bind progress bar update host callback to rng_key split
        rng_expl, rng_train, rng_eval = host_callback.id_tap(
            lambda a, t: pbar.update(1), jax.random.split(rng_loop_iter, 3)
        )

        # exploration rollout
        expl_transitions = eps_greedy_rollout(
            env, dql_state, rng_expl, eps_annealing(dql_state.step), rollout_length
        )
        replay_memory = replay_memory.push(expl_transitions)

        # policy training
        for rng_sample in jax.random.split(rng_train, train_iter):
            train_transitions = replay_memory.sample(rng_sample, rollout_length)
            dql_state = dql_state.update_params_qnet(train_transitions)

        # evaluation rollout
        eval_transitions = eps_greedy_rollout(env, dql_state, rng_eval, 0.0, rollout_length)

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
