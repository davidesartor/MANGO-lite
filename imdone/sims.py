from functools import partial
from typing import Callable, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

from utils import FrozenLake, Transition, ConvNet, RNGKey, ObsType, ActType
from qlearning import DQLTrainState
from replay import CircularBuffer


class Results(NamedTuple):
    eval_reward: jax.Array
    eval_done: jax.Array
    expl_reward: jax.Array
    expl_done: jax.Array


class SimulationState(NamedTuple):
    env: FrozenLake
    dql_state: DQLTrainState
    replay_memory: CircularBuffer


@partial(jax.jit, static_argnames=("map_size", "lr", "replay_capacity"))
def setup_simulation(rng_key, map_size, lr, replay_capacity):
    rng_env, rng_dql = jax.random.split(rng_key)
    env = FrozenLake.make_preset(rng_env, (map_size, map_size))

    env_state, obs = env.reset(rng_key)
    action = env.action_space.sample(rng_key)
    sample_transition = Transition(env_state, obs, action, 0.0, obs, False, {})

    qnet = ConvNet(hidden=[2 * map_size] * int(np.log2(map_size)), out=4)
    dql_state = DQLTrainState.create(rng_dql, qnet, obs, lr)
    replay_memory = CircularBuffer.create(sample_transition, replay_capacity)
    return SimulationState(env, dql_state, replay_memory)


def rollout(
    get_action_fn: Callable[[RNGKey, ObsType], ActType],
    env: FrozenLake,
    rng_key: RNGKey,
    steps: int,
):
    def scan_compatible_step(carry, rng_key: RNGKey):
        env_state, obs = carry
        rng_action, rng_step, rng_reset = jax.random.split(rng_key, 3)
        action = get_action_fn(rng_action, obs)
        next_env_state, next_obs, reward, done, info = env.step(env_state, rng_step, action)
        transition = Transition(env_state, obs, action, reward, next_obs, done, info)

        # reset the environment if done
        next_env_state, next_obs = jax.lax.cond(
            done, lambda: env.reset(rng_reset), lambda: (next_env_state, next_obs)
        )
        return (next_env_state, next_obs), transition

    rng_env_reset, rng_steps = jax.random.split(rng_key)
    rng_steps = jax.random.split(rng_steps, steps)
    env_state, obs = env.reset(rng_env_reset)
    _, transitions = jax.lax.scan(scan_compatible_step, (env_state, obs), rng_steps)
    return transitions


@partial(jax.jit, static_argnames=("steps",))
def greedy_rollout(env: FrozenLake, dql_state: DQLTrainState, rng_key: RNGKey, steps: int):
    def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
        qval = dql_state.qval_apply_fn(dql_state.params_qnet, obs)
        return qval.argmax()

    return rollout(get_action, env, rng_key, steps)


@partial(jax.jit, static_argnames=("steps",))
def random_rollout(env: FrozenLake, dql_state: DQLTrainState, rng_key: RNGKey, steps: int):
    def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
        qval = dql_state.qval_apply_fn(dql_state.params_qnet, obs)
        action = jax.random.choice(rng_key, qval.size)
        return action

    return rollout(get_action, env, rng_key, steps)


@partial(jax.jit, donate_argnames=("sim_state",), static_argnames=("rollout_length", "train_iter"))
def q_learning_step(sim_state: SimulationState, rng_key, rollout_length: int, train_iter: int):
    (env, dql_state, replay_memory) = sim_state
    rng_expl, rng_train, rng_eval = jax.random.split(rng_key, 3)

    # exploration rollout
    exploration = random_rollout(env, dql_state, rng_expl, rollout_length)
    replay_memory = replay_memory.push(exploration)

    # # policy training
    for rng_sample in jax.random.split(rng_train, train_iter):
        transitions = replay_memory.sample(rng_sample, rollout_length)
        dql_state = dql_state.update_params(transitions)

    # evaluation rollout
    evaluation = greedy_rollout(env, dql_state, rng_eval, rollout_length)

    # log results
    results = Results(exploration.reward, exploration.done, evaluation.reward, evaluation.done)
    return SimulationState(env, dql_state, replay_memory), results
