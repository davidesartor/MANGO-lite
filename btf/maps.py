from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def is_connected(frozen: jax.Array):
    def while_cond(carry):
        visited, to_visit = carry
        return to_visit.sum() > 0

    def while_body(carry):
        visited, to_visit = carry
        visited = visited | to_visit

        adj = jnp.pad(to_visit, 1)
        adj = adj[:-2, 1:-1] | adj[2:, 1:-1] | adj[1:-1, :-2] | adj[1:-1, 2:]
        to_visit = adj & frozen & ~visited
        return visited, to_visit

    visited = jnp.zeros_like(frozen)
    start = jnp.unravel_index(frozen.argmax(), frozen.shape)
    to_visit = jnp.zeros_like(frozen).at[start].set(True)
    visited, _ = jax.lax.while_loop(while_cond, while_body, (visited, to_visit))
    return (visited & frozen).sum() == frozen.sum()


@partial(jax.jit, static_argnames=("shape"))
def generate_map(rng_key, shape: tuple[int, int], p: float) -> jax.Array:
    rows, cols = shape
    if rows == 1 and cols == 1:
        return jax.random.uniform(rng_key) < p

    def while_cond(carry):
        rng_key, frozen = carry
        return ~is_connected(frozen)

    def while_body(carry):
        rng_key, frozen = carry
        rng_key, *subkeys = jax.random.split(rng_key, 5)
        chunks = [generate_map(rng_chunk, (rows // 2, cols // 2), p) for rng_chunk in subkeys]
        frozen = jnp.block([[chunks[0], chunks[1]], [chunks[2], chunks[3]]])
        return rng_key, frozen

    _, frozen = jax.lax.while_loop(while_cond, while_body, while_body((rng_key, None)))
    return frozen


@partial(jax.jit, static_argnames=("shape"))
def get_preset_map(shape: tuple[int, int]) -> jax.Array:
    rows, cols = shape
    if rows == cols == 2:
        map = ["FH", "FF"]
    elif rows == cols == 4:
        map = ["FFFF", "FHFH", "FFFH", "HFFF"]
    elif rows == cols == 8:
        map = [
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFF",
        ]
    else:
        raise ValueError(f"no preset map for {rows}x{cols}")
    return jnp.array([[c == "F" for c in row] for row in map])
