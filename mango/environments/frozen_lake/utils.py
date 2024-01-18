from typing import Optional
import numpy as np
import numpy.typing as npt
import numba


@numba.njit
def connected_components(frozen: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """get a boolean mask (1=frozen, 0=lake) and return a tensor of the same shape
    with each element being the index of the connected component the cell belongs to"""
    if frozen.sum() == 0:
        return np.zeros_like(frozen, dtype=np.int_)
    if frozen.sum() == frozen.size:
        return np.ones_like(frozen, dtype=np.int_)

    components = np.arange(1, frozen.size + 1, dtype=np.int_).reshape(frozen.shape)
    components = components * frozen
    changes = np.ones_like(components, dtype=np.bool_)
    while changes.sum() > 0:
        adj = components.copy()
        adj[1:, :] = np.maximum(adj[1:, :], components[:-1, :])
        adj[:-1, :] = np.maximum(adj[:-1, :], components[1:, :])
        adj[:, 1:] = np.maximum(adj[:, 1:], components[:, :-1])
        adj[:, :-1] = np.maximum(adj[:, :-1], components[:, 1:])
        changes = adj * frozen > components * frozen
        components = adj * frozen
    for i, u in enumerate(np.unique(components)):
        components = np.where(components == u, i, components)
    return components


# @numba.njit
def generate(
    log2size: int, p_frozen: float, np_rng=np.random.default_rng()
) -> npt.NDArray[np.bool_]:
    """generate a map of size 2**log2size x 2**log2size"""
    n = 2**log2size
    if n == 1:
        return np.ones((1, 1), dtype=np.bool_) * np_rng.random() < p_frozen

    tile = np.zeros((n, n), dtype=np.bool_)

    tile[: n // 2, : n // 2] = generate(log2size - 1, p_frozen, np_rng)
    tile[n // 2 :, : n // 2] = generate(log2size - 1, p_frozen, np_rng)
    tile[: n // 2, n // 2 :] = generate(log2size - 1, p_frozen, np_rng)
    tile[n // 2 :, n // 2 :] = generate(log2size - 1, p_frozen, np_rng)
    while connected_components(tile).max() > 1:
        rand = np_rng.integers(0, 4)
        if rand == 0:
            tile[: n // 2, : n // 2] = generate(log2size - 1, p_frozen, np_rng)
        elif rand == 1:
            tile[n // 2 :, : n // 2] = generate(log2size - 1, p_frozen, np_rng)
        elif rand == 2:
            tile[: n // 2, n // 2 :] = generate(log2size - 1, p_frozen, np_rng)
        elif rand == 3:
            tile[n // 2 :, n // 2 :] = generate(log2size - 1, p_frozen, np_rng)
    return tile


@numba.njit
def sample_position_in(
    region: npt.NDArray[np.bool_],
    avoid: Optional[tuple[int, int]] = None,
    np_rng=np.random.default_rng(),
) -> tuple[int, int]:
    if avoid is not None:
        region[avoid] = False
    if region.sum() == 0:
        raise ValueError("No position available")
    r, c = np_rng.integers(0, region.shape[0]), np_rng.integers(0, region.shape[1])
    while not region[r, c]:
        r, c = np_rng.integers(0, region.shape[0]), np_rng.integers(0, region.shape[1])
    return r, c


def generate_map(
    shape: tuple[int, int] = (8, 8),
    p: float | None = None,
    seed: Optional[int] = None,
):
    np_rng = np.random.default_rng(seed)
    connected = generate(
        log2size=int(np.log2(shape[0])),
        p_frozen=p or np_rng.uniform(0.5, 1.0),
        np_rng=np_rng,
    )
    start_pos = sample_position_in(connected, np_rng=np_rng)
    goal_pos = sample_position_in(connected, avoid=start_pos, np_rng=np_rng)

    desc = np.empty(shape, dtype="U1")
    desc[connected] = b"F"
    desc[~connected] = b"H"
    desc[start_pos] = "S"
    desc[goal_pos] = "G"
    desc = ["".join(row) for row in desc]
    return desc
