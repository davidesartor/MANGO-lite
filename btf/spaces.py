from typing import Any, Protocol, TypeVar
import jax
import jax.numpy as jnp
from flax import struct


class Space(Protocol):
    shape: tuple[int, ...]
    dtype: jnp.dtype

    def sample(self, rng: jax.Array) -> Any:
        ...

    def contains(self, x) -> jnp.bool_:
        ...


@struct.dataclass
class Discrete(Space):
    n: int
    shape: tuple[int, ...] = ()
    dtype: jnp.dtype = jnp.int32

    def sample(self, rng: jax.Array) -> jnp.int64:
        return jax.random.randint(rng, shape=(), minval=0, maxval=self.n)

    def contains(self, x: jnp.int_) -> jnp.bool_:
        return (x >= 0) & (x < self.n)
