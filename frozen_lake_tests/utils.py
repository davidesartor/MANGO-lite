import matplotlib.pyplot as plt

from mango.environments import frozen_lake


def render(
    env: frozen_lake.wrappers.FrozenLakeWrapper,
    title=f"Environment",
    figsize=None,
):
    if figsize:
        plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.imshow(env.unwrapped.render())
    plt.xticks([])
    plt.yticks([])
    plt.show()
