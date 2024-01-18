import os
import sys
import torch
from tqdm import tqdm
import numpy as np

from mango.environments import frozen_lake

cuda_idx = 2
device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else "cpu")

# make environment
env_params = dict(
    map_name="RANDOM", p=None, shape=(4, 4), start_pos=[(0, 0)], goal_pos=[(-1, -1)], seed=0
)
env = frozen_lake.CustomFrozenLakeEnv()
env = frozen_lake.wrappers.ReInitOnReset(env, **params)
env = frozen_lake.wrappers.TensorObservation(env, one_hot=True)


def make_env(
    map_base: int, map_scale: int, p_frozen: float | None, one_shot, seed: Optional[int] = None
):
    return env


# create the environment and the agent
env = utils_sim.make_env(map_base, map_scale, p_frozen, one_shot, seed=run_id)
if use_mango:
    agent = utils_sim.make_mango_agent(env, map_base, map_scale, mask_state, device=device)
else:
    agent = utils_sim.make_agent(env, map_base, map_scale, device=device)

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
