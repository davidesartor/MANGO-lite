from functools import partial
from typing import Callable, Sequence
from flax import linen as nn, struct
import jax
import jax.numpy as jnp

from frozen_lake import FrozenLake, EnvState, ObsType, ActType, RNGKey


class Transition(struct.PyTreeNode):
    env_state: EnvState
    obs: ObsType
    action: ActType
    reward: float
    next_obs: ObsType
    done: bool
    info: dict


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
def random_rollout(env: FrozenLake, rng_key: RNGKey, steps: int):
    def get_action(rng_key: RNGKey, obs: ObsType) -> ActType:
        action = env.action_space.sample(rng_key)
        return action

    return rollout(get_action, env, rng_key, steps)


class ConvNet(nn.Module):
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x):
        for ch in self.hidden:
            x = nn.Conv(ch, (3, 3))(x)
            x = nn.celu(x)
            x = nn.Conv(ch, (2, 2), strides=(2, 2))(x)
            x = nn.LayerNorm()(x)
        x = x.flatten()
        x = nn.Dense(features=self.out)(x)
        return x


class OuterPolicyQnet(nn.Module):
    map_scale: int
    actions: int

    @nn.compact
    def __call__(self, x):
        x = ConvNet(hidden=[32] * self.map_scale, out=self.actions)(x)
        return x


@jax.jit
def grid_coord(obs: ObsType, cell_size: jax.Array) -> jax.Array:
    row, col, cha = obs.shape
    agent_idx = obs[:, :, 0].argmax()
    coord = jnp.array(divmod(agent_idx, col))
    return coord // cell_size


class InnerPolicyQnet(nn.Module):
    cell_scale: int
    comands: int
    actions: int
    mask: bool = False

    @nn.compact
    def __call__(self, x):
        rows, cols, chan = x.shape
        if self.mask:
            cell_size = 2**self.cell_scale
            slice_start = grid_coord(x, cell_size) * cell_size
            x = jax.lax.dynamic_slice(
                jnp.pad(x, 1), (*slice_start, 1), (cell_size + 2, cell_size + 2, chan)
            )
        MultiConvNet = nn.vmap(
            ConvNet,
            in_axes=None,  # type: ignore
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.comands,
        )
        x = MultiConvNet(hidden=[128] * self.cell_scale, out=self.actions)(x)
        return x
