# %%
import sys

sys.path.append("..")

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from mango.mango import Mango
from mango.utils import plot_loss_reward
from mango.environments import frozen_lake
from mango.actions import grid2d

# %%
init_kwargs = dict(map_name="RANDOM", shape=(16, 16), p=0.5, goal_pos=None, start_pos=None)
env = frozen_lake.CustomFrozenLakeEnv(**init_kwargs)  # type: ignore
env = frozen_lake.ReInitOnReset(env, **init_kwargs)
env = frozen_lake.TensorObservation(env, one_hot=True)

env_shape = (env.unwrapped.ncol, env.unwrapped.nrow)  # type: ignore
obs, _ = env.reset()
print(f"{obs.shape=}, {obs.dtype=}")
plt.imshow(env.render())  # type: ignore
plt.show()

# %%
n_layers = 3
gamma = 0.8
mango = Mango(
    environment=env,
    abstract_actions=[
        grid2d.SubGridMovementOnehot(
            cell_shape=(2**i, 2**i),
            grid_shape=env_shape,
            agent_channel=0,
            add_valid_channel=True,
            reward=(1 - gamma),
            p_termination=0.1,
        )
        for i in range(1, n_layers)
    ],
    policy_params=dict(
        lr=1e-2,
        gamma=gamma,
        net_params=dict(
            hidden_channels=[8, 8],
            hidden_features=[64, 16],
            activation_out=torch.nn.Tanh(),
            batch_norm=True,
            residual_connections=False,
            #device=torch.device("cuda") if torch.cuda.is_available() else None,
        ),
    ),
)
print(mango)

# %%
import snakeviz
%load_ext snakeviz

# %%
%%snakeviz -t
N = 300
for layer in [1, 2]:
    for randomness in (pbar := tqdm(np.linspace(1.0, 0.0, N))):
        pbar.set_description(f"randomness: {randomness:.3f}")
        mango.set_randomness(randomness, layer=layer)
        obs, reward, term, trunc, info = mango.explore(layer=layer, episode_length=2)
        mango.train(layer=layer)

# %%
# mango.save_to("trained_models/frozen_lake_mango.pickle")

# %%
plot_loss_reward(mango, grid2d.Actions)
frozen_lake.plot_all_qvals(mango)
plt.show()

# %%
obs, info = mango.reset()
frozen_lake.plot_all_qvals(mango)
plt.show()

trajectory = [obs]
for action in grid2d.Actions:
    print(action.name)
    for step in range(10):
        obs, reward, trunc, term, info = mango.step((-1, action.value))
        trajectory.extend(info["mango:trajectory"])
        if not info["mango:truncated"]:
            break
    frozen_lake.plot_all_abstractions(mango, trajectory)
    plt.show()


