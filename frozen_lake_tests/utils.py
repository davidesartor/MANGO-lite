import matplotlib.pyplot as plt
import numpy as np

from mango.environments import frozen_lake
from mango.protocols import ObsType, Transition


def smooth(signal, window=0.01):
    window = min((1000, max(3, int(len(signal) * window))))
    if len(signal) < 10:
        return signal
    window_array = np.ones(window) / window
    return np.convolve(signal, window_array, mode="valid")


def render(
    env: frozen_lake.wrappers.FrozenLakeWrapper,
    title="Environment",
    trajectory: list[ObsType] = [],
    abstraction_sizes: list[tuple[int, int]] = [],
    figsize=None,
):
    if figsize:
        plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.imshow(env.unwrapped.render())
    plt.xticks([])
    plt.yticks([])

    # draw trajectory
    sy, sx = env.unwrapped.cell_size
    for obs1, obs2 in zip(trajectory[:-1], trajectory[1:]):
        y1, x1 = divmod(env.observation_inv(obs1), env.unwrapped.desc.shape[1])
        y2, x2 = divmod(env.observation_inv(obs2), env.unwrapped.desc.shape[1])
        y1, x1, y2, x2 = int(y1) + 0.5, int(x1) + 0.5, int(y2) + 0.5, int(x2) + 0.5
        plt.plot([x1 * sx, x2 * sx], [y1 * sy, y2 * sy], ":k", linewidth=4)

    # draw abstraction
    for l, cell_size in enumerate(abstraction_sizes):
        grid = tuple((s - 2) // c for s, c in zip(env.unwrapped.desc.shape, cell_size))
        square = tuple(s * c for s, c in zip(env.unwrapped.cell_size, cell_size))
        # draw squares around all cells of the given size
        for i, j in np.ndindex(grid):
            plt.gca().add_patch(
                plt.Rectangle(  # type: ignore
                    (sx + j * square[1], sy + i * square[0]),
                    square[1],
                    square[0],
                    fill=False,
                    edgecolor="red",
                    linewidth=4**l,
                    alpha=0.5,
                )
            )
