from typing import Any, Optional
import numpy as np
import numpy.typing as npt
import numba


@numba.njit
def sample_position_in(region: npt.NDArray[np.bool_]) -> tuple[int, int]:
    r, c = np.random.randint(region.shape[0]), np.random.randint(region.shape[1])
    if region.sum() == 0:
        return r, c
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
    shape: tuple[int, int], p: float, contains: Optional[tuple[int, int]] = None
) -> npt.NDArray[np.bool_]:
    connected = np.zeros(shape, dtype=np.bool_)
    start = sample_position_in(connected) if contains is None else contains
    connected[start] = True
    frozen = np.random.random_sample(shape) < p
    connected |= reachable_from(connected, frozen)
    while connected.sum() < p * connected.size:
        extension = sample_position_in(~(connected + frozen))
        frozen[extension] = True
        connected |= reachable_from(connected, frozen + connected)
    return connected


def generate_map(
    shape=(8, 8),
    p=0.8,
    start_pos: tuple[int, int] | None = (0, 0),
    goal_pos: tuple[int, int] | None = (-1, -1),
    multi_start=False,
    mirror=False,
):
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    if start_pos and goal_pos:
        start_pos = start_pos[0] % shape[0], start_pos[1] % shape[1]
        goal_pos = goal_pos[0] % shape[0], goal_pos[1] % shape[1]
        connected = random_board(shape, p, contains=goal_pos)
        while not connected[start_pos]:
            connected = random_board(shape, p, contains=goal_pos)
    else:
        connected = random_board(shape, p, contains=(goal_pos or start_pos))
        start_pos = start_pos or sample_position_in(connected)
        while not goal_pos or goal_pos == start_pos:
            goal_pos = sample_position_in(connected)

    desc = np.empty(shape, dtype="U1")
    desc[connected] = b"F"
    desc[~connected] = b"H"
    if multi_start:
        desc[connected] = "S"
    else:
        desc[start_pos] = "S"
    desc[goal_pos] = "G"
    desc = ["".join(row) for row in desc]
    if mirror:
        desc = [row[::-1] + row[shape[0] % 2 :] for row in desc[::-1] + desc[shape[1] % 2 :]]
    return desc
