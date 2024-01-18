from typing import Any, Optional, Sequence
import numpy as np
import numpy.typing as npt
import numba
import warnings


@numba.njit
def sample_position_in(
    region: npt.NDArray[np.bool_],
    avoid: Optional[Sequence[tuple[int, int]]] = None,
    np_rng=np.random.default_rng(),
) -> tuple[int, int]:
    if avoid is not None:
        for r, c in avoid:
            region[r, c] = False
    if region.sum() == 0:
        raise ValueError("No position available")
    r, c = np_rng.integers(0, region.shape[0]), np_rng.integers(0, region.shape[1])
    while not region[r, c]:
        r, c = np_rng.integers(0, region.shape[0]), np_rng.integers(0, region.shape[1])
    return r, c


@numba.njit
def reachable_from(
    start: npt.NDArray[np.bool_], frozen: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    reached = start.copy() & frozen
    total = 0
    while total < reached.sum():
        total = reached.sum()
        adj = np.zeros(reached.shape, dtype=np.bool_)
        adj[1:, :] |= reached[:-1, :]
        adj[:-1, :] |= reached[1:, :]
        adj[:, 1:] |= reached[:, :-1]
        adj[:, :-1] |= reached[:, 1:]
        reached |= adj & frozen
    return reached


@numba.njit
def ensure_unanbiguos(connected: npt.NDArray[np.bool_]):
    bad_patterns = [
        np.array([1, 0, 1, 1], dtype=np.bool_),
        np.array([0, 1, 0, 1], dtype=np.bool_),
        np.array([1, 0, 0, 1], dtype=np.bool_),
    ]
    fixed_patterns = [
        np.array([1, 1, 1, 1], dtype=np.bool_),
        np.array([0, 1, 1, 1], dtype=np.bool_),
        np.array([1, 1, 1, 1], dtype=np.bool_),
    ]
    for idx, (min, max) in [(3, (0, 4)), (4, (0, 4)), (3, (4, 8)), (4, (4, 8))]:
        for bad, fixed in zip(bad_patterns, fixed_patterns):
            if (connected[idx, min:max] == bad).all():
                connected[idx, min:max] = fixed
            if (connected[idx, min:max] == bad[::-1]).all():
                connected[idx, min:max] = fixed[::-1]
            if (connected[min:max, idx] == bad).all():
                connected[min:max, idx] = fixed
            if (connected[min:max, idx] == bad[::-1]).all():
                connected[min:max, idx] = fixed[::-1]
    return connected


@numba.njit
def random_board(
    shape: tuple[int, int],
    p: float,
    contains: Optional[Sequence[tuple[int, int]]] = None,
    np_rng=np.random.default_rng(),
) -> npt.NDArray[np.bool_]:
    frozen = np.zeros(shape, dtype=np.bool_)
    need_to_connect = np.zeros(shape, dtype=np.bool_)
    start = np.zeros(shape, dtype=np.bool_)
    if contains is not None and contains:
        start[contains[0]] = True
        for c in contains:
            need_to_connect[c] = True
    else:
        start[sample_position_in(np.ones(shape, dtype=np.bool_), np_rng=np_rng)] = True

    connected = start.copy()
    frozen = np_rng.random(shape) < p
    connected = reachable_from(connected, frozen | connected | need_to_connect)
    while True:
        if (
            connected.sum() < (p - 0.05) * connected.size
            or (need_to_connect & ~connected).sum() > 0
        ):
            expansion = sample_position_in(~(connected | frozen | need_to_connect), np_rng=np_rng)
            frozen[expansion] = True
            connected = reachable_from(connected, connected | frozen | need_to_connect)
        elif connected.sum() > (p + 0.05) * connected.size:
            contraction = sample_position_in(connected & ~need_to_connect, np_rng=np_rng)
            frozen[contraction] = False
            connected = reachable_from(start, frozen | need_to_connect)
        else:
            break
    return connected


def generate_map(
    shape: tuple[int, int] = (8, 8),
    p: float | None = None,
    start_pos: Sequence[tuple[int, int]] = [],
    goal_pos: Sequence[tuple[int, int]] = [],
    start_anywhere=False,
    mirror=False,
    seed: Optional[int] = None,
):
    np_rng = np.random.default_rng(seed)
    if p is None:
        p = np.random.uniform(0.3, 1.0)

    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")

    start_pos = [(r % shape[0], c % shape[1]) for r, c in start_pos]
    goal_pos = [(r % shape[0], c % shape[1]) for r, c in goal_pos]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if goal_pos and start_pos:
            start_pos_idx = np_rng.integers(len(start_pos))
            goal_pos_idx = np_rng.integers(len(goal_pos))
            while start_pos[start_pos_idx] == goal_pos[goal_pos_idx]:
                start_pos_idx = np_rng.integers(len(start_pos))
                goal_pos_idx = np_rng.integers(len(goal_pos))
            goal_pos = [goal_pos[goal_pos_idx]]
            start_pos = [start_pos[start_pos_idx]]
            connected = random_board(shape, p, contains=goal_pos + start_pos, np_rng=np_rng)
        elif start_pos and not goal_pos:
            start_pos = [start_pos[np_rng.integers(len(start_pos))]]
            connected = random_board(shape, p, contains=start_pos, np_rng=np_rng)
            goal_pos = [sample_position_in(connected, avoid=start_pos, np_rng=np_rng)]
        elif not start_pos and goal_pos:
            goal_pos = [goal_pos[np_rng.integers(len(goal_pos))]]
            connected = random_board(shape, p, contains=goal_pos, np_rng=np_rng)
            start_pos = [sample_position_in(connected, avoid=goal_pos, np_rng=np_rng)]
        else:
            connected = random_board(shape, p, contains=None, np_rng=np_rng)
            start_pos = [sample_position_in(connected, np_rng=np_rng)]
            goal_pos = [sample_position_in(connected, avoid=start_pos, np_rng=np_rng)]

    # TODO: do this properly for general abstractions
    if connected.shape == (8, 8):
        connected = ensure_unanbiguos(connected)

    desc = np.empty(shape, dtype="U1")
    desc[connected] = b"F"
    desc[~connected] = b"H"
    if start_anywhere:
        desc[connected] = "S"
    for r, c in start_pos:
        desc[r, c] = "S"
    for r, c in goal_pos:
        desc[r, c] = "G"
    desc = ["".join(row) for row in desc]
    if mirror:
        desc = [row[::-1] + row[shape[0] % 2 :] for row in desc[::-1] + desc[shape[1] % 2 :]]
    return desc
