from typing import Any, Optional, Sequence
import numpy as np
import numpy.typing as npt
import numba
import warnings


@numba.njit
def sample_position_in(
    region: npt.NDArray[np.bool_], avoid: Optional[Sequence[tuple[int, int]]] = None
) -> tuple[int, int]:
    if avoid is not None:
        for r, c in avoid:
            region[r, c] = False
    if region.sum() == 0:
        raise ValueError("No position available")
    r, c = np.random.randint(region.shape[0]), np.random.randint(region.shape[1])
    while not region[r, c]:
        r, c = np.random.randint(region.shape[0]), np.random.randint(region.shape[1])
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
def random_board(
    shape: tuple[int, int], p: float, contains: Optional[Sequence[tuple[int, int]]] = None
) -> npt.NDArray[np.bool_]:
    connected = np.zeros(shape, dtype=np.bool_)
    need_to_connect = np.zeros(shape, dtype=np.bool_)
    if contains is not None and contains:
        start = contains[0]
        for c in contains:
            need_to_connect[c] = True
        need_to_connect[start] = False
    else:
        start = sample_position_in(~connected)

    connected[start] = True
    frozen = np.random.random_sample(shape) < p
    frozen |= need_to_connect
    connected |= reachable_from(connected, frozen)
    while connected.sum() < p * connected.size or need_to_connect.sum() > 0:
        extension = sample_position_in(~(connected + frozen))
        frozen[extension] = True
        connected |= reachable_from(connected, frozen + connected)
        need_to_connect &= ~connected
    return connected


def one_hot_encode(
    coord: Sequence[tuple[int, int]], grid_shape: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    grid = np.zeros(grid_shape, dtype=np.bool_)
    for r, c in coord:
        grid[r, c] = True
    return grid


def generate_map(
    shape: tuple[int, int] = (8, 8),
    p: float = 0.8,
    start_pos: Sequence[tuple[int, int]] = [],
    goal_pos: Sequence[tuple[int, int]] = [],
    start_anywhere=False,
    mirror=False,
):
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")

    start_pos = [(r % shape[0], c % shape[1]) for r, c in start_pos]
    goal_pos = [(r % shape[0], c % shape[1]) for r, c in goal_pos]
    if len(goal_pos) > 1:
        goal_pos = [goal_pos[np.random.randint(len(goal_pos))]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if start_pos and goal_pos:
            connected = random_board(shape, p, contains=goal_pos + start_pos)
        elif start_pos and not goal_pos:
            connected = random_board(shape, p, contains=start_pos)
            goal_pos = [sample_position_in(connected, avoid=start_pos)]
        elif not start_pos and goal_pos:
            connected = random_board(shape, p, contains=goal_pos)
            start_pos = [sample_position_in(connected, avoid=goal_pos)]
        else:
            connected = random_board(shape, p, contains=None)
            start_pos = [sample_position_in(connected)]
            goal_pos = [sample_position_in(connected, avoid=start_pos)]

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
