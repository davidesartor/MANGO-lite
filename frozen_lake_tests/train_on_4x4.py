import os
import sys
from typing import Any
import torch
from tqdm import tqdm
import numpy as np

from mango.environments import frozen_lake
from mango.mango import Mango
from mango.policies.dqnet import DQNetPolicy

cuda_idx = 2
device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else "cpu")
run_id = 0

lr = 1e-3
gamma = 0.95


# make environment
env_params: dict[str, Any] = dict(map_name="RANDOM", p=None, shape=(4, 4), seed=run_id)
env = frozen_lake.CustomFrozenLakeEnv(**env_params)
env = frozen_lake.wrappers.ReInitOnReset(env, **env_params)
env = frozen_lake.wrappers.TensorObservation(env, one_hot=True)


def net_params(conv_filters: int, field_size: int, device) -> dict[str, Any]:
    """Get the params a minimal conv network with a given receptive field size."""
    n_layers = 2 * int(np.log2(field_size)) - 1  # ConvEncoder pools by 2 every other layer
    return dict(
        hidden_channels=[conv_filters] * n_layers,
        hidden_features=[],
        device=device,
    )


policy_params = [
    dict(lr=1e-3, gamma=0.95, net_params=net_params(conv_filters=8, field_size=4, device=device))
    for layer in range(2)
]


dict(policy_cls=DQNetPolicy, policy_params=policy_params(map_base, map_scale, lr, gamma, device))

mango_agent = Mango(
    environment=env,
    abstract_actions=abstract_actions(map_base, map_scale, cell_scales, mask_state),
    policy_cls=DQNetPolicy,
    policy_params=dict(
        lr=lr,
        gamma=gamma,
        net_params=net_params(conv_filters=8, field_size=4, device=device),
    ),
    dynamic_policy_params=[
        dynamic_policy_params(map_base, scale, lr, gamma, device) for scale in cell_scales
    ],
)
return mango_agent


# train loop
annealing_episodes, max_episodes, train_steps, episode_length = utils_sim.train_params(
    map_base, map_scale, p_frozen, one_shot
)
p_bar_descr = "training " + ("mango_agent" if use_mango else "normal_agent")
# if not use_mango:
#     max_episodes *= 5
randomness = np.concatenate(
    [np.linspace(1.0, 0.2, annealing_episodes), np.ones(max_episodes) * 0.2]
)
episode_rewards = [0.0] * solved_treshold[1]
for r in tqdm(randomness, desc=p_bar_descr, leave=False):
    # exploration
    agent.run_episode(randomness=r, episode_length=episode_length)

    # training
    for _ in range(train_steps):
        agent.train()

    # evaluation
    trajectory, rewards = agent.run_episode(randomness=0.0, episode_length=episode_length)

    # early stopping when the agent is good enough
    episode_rewards.append(np.sum(rewards))
    if float(np.mean(episode_rewards[-solved_treshold[1] :])) >= solved_treshold[0]:
        break
