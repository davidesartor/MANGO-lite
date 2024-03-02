from functools import partial
from typing import Callable, Generic, TypeVar
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
from flax import struct
from frozen_lake import Env, ObsType, ActType, RNGKey, Transition


def rollout(
    get_action_fn: Callable[[RNGKey, ObsType], ActType],
    env: Env,
    rng_key: RNGKey,
    steps: int,
    pbar_desc: str = "Rollout",
):
    pbar = tqdm(total=steps, desc=pbar_desc)

    def scan_compatible_step(carry, rng_key: RNGKey):
        env_state, obs = host_callback.id_tap(lambda a, t: pbar.update(1), carry)

        rng_action, rng_step, rng_reset = jax.random.split(rng_key, 3)
        action = get_action_fn(rng_action, obs)
        next_env_state, next_obs, reward, done, info = env.step(env_state, rng_step, action)
        transition = Transition(env_state, obs, action, reward, next_obs, done, info)

        # reset the environment if done
        (next_env_state, next_obs) = jax.lax.cond(
            done, lambda: env.reset(rng_reset), lambda: (next_env_state, next_obs)
        )
        return (next_env_state, next_obs), transition

    rng_init, rng_steps = jax.random.split(rng_key)
    rng_steps = jax.random.split(rng_steps, steps)
    env_state, obs = env.reset(rng_init)
    _, transitions = jax.lax.scan(scan_compatible_step, (env_state, obs), rng_steps)
    return transitions


@partial(jax.jit, static_argnames=("steps",))
def random_rollout(
    env: Env,
    rng_key: RNGKey,
    steps: int,
):
    get_action_fn = lambda rng_key, obs: env.action_space.sample(rng_key)
    return rollout(get_action_fn, env, rng_key, steps, pbar_desc="Random Rollout")


@partial(jax.jit, static_argnames=("steps", "n_rollouts"))
def multi_random_rollout(
    env: Env,
    rng_key: RNGKey,
    steps: int,
    n_rollouts: int,
):
    rng_keys = jax.random.split(rng_key, n_rollouts)
    return jax.vmap(random_rollout, in_axes=(None, 0, None))(env, rng_keys, steps)


ElementType = TypeVar("ElementType", bound=struct.PyTreeNode)


class CircularBuffer(struct.PyTreeNode, Generic[ElementType]):
    stored_elements: ElementType
    capacity: int = struct.field(pytree_node=False)
    size: int = 0
    last: int = -1

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "capacity"))
    def create(cls, sample: ElementType, capacity: int):
        memory = jax.tree_map(lambda x: jnp.zeros((capacity, *x.shape), x.dtype), sample)
        return cls(memory, capacity)

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def store_episodes(cls, episodes: ElementType):
        transitions = jax.tree_map(lambda x: jnp.concatenate(x, axis=0), episodes)
        n_items, _ = jax.tree_flatten(jax.tree_map(lambda x: x.shape[0], transitions))
        return cls(transitions, capacity=n_items[0], size=n_items[0])

    @partial(jax.jit, donate_argnames=("self",))
    def extend(self, samples: ElementType):
        n_items, _ = jax.tree_flatten(jax.tree_map(lambda x: x.shape[0], samples))

        def update_circular_buffer(mem, elem):
            idxs = (jnp.arange(n_items[0]) + self.last + 1) % self.capacity
            mem = mem.at[idxs].set(elem)
            return mem

        new_size = self.size + n_items[0]
        new_size = jax.lax.select(self.capacity < new_size, self.capacity, new_size)
        new_last = (self.last + n_items[0]) % self.capacity
        new_stored_elements = jax.tree_map(update_circular_buffer, self.stored_elements, samples)
        return self.replace(stored_elements=new_stored_elements, last=new_last, size=new_size)

    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(self, rng_key: jax.Array, batch_size: int) -> ElementType:
        idxs = jax.random.randint(rng_key, shape=(batch_size,), minval=0, maxval=self.size)
        return jax.tree_map(lambda x: x[idxs], self.stored_elements)
