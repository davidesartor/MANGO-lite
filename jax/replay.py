from __future__ import annotations
from functools import partial, wraps
from typing import Callable, Generic, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
import optax

from utils import FrozenLake, ObsType, ActType, RNGKey, Transition


class CircularBuffer(struct.PyTreeNode):
    transitions: Transition
    capacity: int
    last: int = -1
    size: int = 0

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "capacity"))
    def create(cls, sample: Transition, capacity: int):
        memory = jax.tree_map(lambda x: jnp.zeros((capacity, *x.shape[1:]), x.dtype), sample)
        return cls(memory, capacity)

    @partial(jax.jit, donate_argnames=("self",))
    def push(self, transition: Transition):
        n_items = jax.tree_flatten(transition)[0][0].shape[0]
        assert [n_items == x.shape[0] for x in jax.tree_flatten(transition)[0]]

        def update_circular_buffer(mem, elem):
            idxs = (jnp.arange(n_items) + self.last + 1) % self.capacity
            mem = mem.at[idxs].set(elem)
            return mem

        new_size = self.size + n_items
        new_size = jax.lax.select(self.capacity < new_size, self.capacity, new_size)
        new_last = (self.last + n_items) % self.capacity
        new_trans_mem = jax.tree_map(update_circular_buffer, self.transitions, transition)
        return self.replace(transitions=new_trans_mem, last=new_last, size=new_size)

    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(self, rng_key: jax.Array, batch_size: int) -> Transition:
        idxs = jax.random.randint(rng_key, shape=(batch_size,), minval=0, maxval=self.size)
        return jax.tree_map(lambda x: x[idxs], self.transitions)
