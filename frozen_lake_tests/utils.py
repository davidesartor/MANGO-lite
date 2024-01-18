import matplotlib.pyplot as plt
import numpy as np

from mango.environments import frozen_lake


def render(
    env: frozen_lake.wrappers.FrozenLakeWrapper,
    title="Environment",
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
    for l, cell_size in enumerate(abstraction_sizes):
        grid_shape = env.unwrapped.desc.shape
        if env.unwrapped.fail_on_out_of_bounds:
            grid_shape = tuple(s - 2 for s in grid_shape)
        grid_shape = tuple(s // c for s, c in zip(grid_shape, cell_size))
        square = tuple(s * c for s, c in zip(env.unwrapped.cell_size, cell_size))
        # draw squares around all cells of the given size
        for i, j in np.ndindex(grid_shape):
            plt.gca().add_patch(
                plt.Rectangle(  # type: ignore
                    (j * square[1], i * square[0]),
                    square[1],
                    square[0],
                    fill=False,
                    edgecolor="red",
                    linewidth=4**l,
                    alpha=0.5,
                )
            )
