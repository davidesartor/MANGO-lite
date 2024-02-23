from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
from tqdm.auto import tqdm
from frozen_lake import Transition, ObsType, RNGKey

import qlearning
import nets
import mangoenv
import utils


def grid_coord(obs: ObsType, cell_scale: int) -> jax.Array:
    # row, col, cha = obs.shape
    # agent_idx = obs[:, :, 0].argmax()
    # coord = jnp.array(divmod(agent_idx, col))
    coord = obs[0]
    return coord // 2**cell_scale


def beta_fn(cell_scale: int, transition: Transition) -> jnp.bool_:
    cell_start = grid_coord(transition.obs, cell_scale)
    cell_end = grid_coord(transition.next_obs, cell_scale)
    stop_cond = (cell_start != cell_end).any() | transition.done
    return stop_cond


def reward_fn(cell_scale: int, transition: Transition) -> jax.Array:
    cell_start = grid_coord(transition.obs, cell_scale)
    cell_end = grid_coord(transition.next_obs, cell_scale)
    dy, dx = (cell_end - cell_start).astype(float)
    reward = jnp.array((-dx, dy, dx, -dy, transition.reward))
    reward = jax.lax.select(transition.done, reward.at[:-1].set(0.0), reward)
    return reward


@partial(jax.jit, static_argnames=("map_scale", "cell_scale"))
def setup_stage(
    map_scale: int,
    cell_scale: int,
    lower_stage: mangoenv.MangoEnv,
    lr: float,
    rng_key: RNGKey,
):
    qnet = nets.InnerQnet(lower_stage.action_space.n, map_shape=(2**map_scale, 2**map_scale))
    rew_fn = partial(reward_fn, cell_scale)
    b_fn = partial(beta_fn, cell_scale)
    dql_state = qlearning.MultiDQLTrainState.create(
        rng_key, qnet, lower_stage, reward_fn=rew_fn, beta_fn=b_fn, lr=lr
    )
    return dql_state


def train_stage(
    dql_state,
    lower_stage: mangoenv.MangoEnv,
    rng_steps: RNGKey,
    rollout_steps: int,
    eval_steps: int,
    batch_size: int,
):
    pbar = tqdm(range(len(rng_steps)))

    def sim_step(dql_state, rng_key):
        rng_expl, rng_train, rng_eval = host_callback.id_tap(
            lambda a, t: pbar.update(1), jax.random.split(rng_key, 3)
        )
        transitions = utils.random_rollout(lower_stage, rng_expl, rollout_steps)
        dql_state = dql_state.update_replay(transitions)

        transitions = dql_state.replay_buffer.sample(rng_train, batch_size)
        transitions = dql_state.process_transitions(transitions)
        dql_state = dql_state.update_params(transitions)

        transitions = dql_state.greedy_rollout(lower_stage, rng_eval, eval_steps)
        return dql_state, (transitions.reward, transitions.done)

    dql_state, (rewards, dones) = jax.lax.scan(sim_step, dql_state, rng_steps)
    return dql_state, rewards, dones
