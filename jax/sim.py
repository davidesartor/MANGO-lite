from typing import Callable
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
from tqdm.auto import tqdm

import frozen_lake
import utils
import replay
import qlearning


def simulation_setup(rng_key: jax.Array, map_size):
    rng_env, rng_qnet = jax.random.split(rng_key)

    env = frozen_lake.FrozenLake.make(rng_env, (map_size, map_size))

    qnet = utils.ConvNet(hidden=[2 * map_size] * int(jnp.log2(map_size)), out=4)
    env_state, env_obs = env.reset(rng_env)  # rng proxy, only used for shape
    qnet_params = qnet.init(rng_qnet, env_obs)

    dql_state = qlearning.DDQNTrainState.create(apply_fn=qnet.apply, params=qnet_params)
    rollout = utils.RolloutManager(env, qnet.apply)

    init_transitions = rollout.eps_greedy(dql_state.params, rng_key, 0.0, 2)
    replay_memory = replay.CircularBuffer.create(init_transitions)
    return dql_state, replay_memory, rollout


def run_q_learning_simulation(
    rng_key: jax.Array,
    map_size: int,
    n_rollouts: int,
    rollout_length: int,
    train_iter: int,
    eps_annealing: Callable[[int], float],
):
    rng_init, rng_loop = jax.random.split(rng_key)
    dql_state, replay_memory, rollout = simulation_setup(rng_init, map_size)

    def loop_body(carry, rng_key):
        dql_state, replay_memory = carry
        # bind progress bar update host callback to rng_key split
        rng_expl, rng_train, rng_eval = host_callback.id_tap(
            lambda a, t: pbar.update(1), jax.random.split(rng_key, 3)
        )

        # exploration rollout
        expl_transitions = rollout.eps_greedy(
            dql_state.params,
            rng_key=rng_expl,
            epsilon=eps_annealing(dql_state.step),
            n_steps=rollout_length,
        )
        replay_memory = replay_memory.push(expl_transitions)

        # policy training
        for rng_sample in jax.random.split(rng_train, train_iter):
            train_transitions = replay_memory.sample(rng_sample, rollout_length)
            dql_state = dql_state.update(train_transitions)

        # evaluation rollout
        eval_transitions = rollout.eps_greedy(
            dql_state.params,
            rng_key=rng_eval,
            epsilon=0.0,
            n_steps=rollout_length,
        )
        return (dql_state, replay_memory), (eval_transitions.reward, eval_transitions.done)

    pbar = tqdm(total=n_rollouts)
    (dql_state, replay_memory), eval = jax.lax.scan(
        f=loop_body,
        init=(dql_state, replay_memory),
        xs=jax.random.split(rng_loop, n_rollouts),
    )
    pbar.close()
    return dql_state, eval
