from functools import partial
from typing import Callable, Sequence
from flax import linen as nn, struct
import jax

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
    out: int | Sequence[int]

    @nn.compact
    def __call__(self, x):
        for ch in self.hidden:
            x = nn.Conv(ch, (3, 3))(x)
            x = nn.celu(x)
            x = nn.Conv(ch, (2, 2), strides=(2, 2))(x)
            x = nn.LayerNorm()(x)
        x = x.flatten()
        x = nn.DenseGeneral(features=self.out)(x)
        return x
