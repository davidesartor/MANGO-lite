import os
import sys
import torch
from tqdm import tqdm
import numpy as np

# parameters for the environment
map_scale = 3
p_frozen = 0.5
one_shot = True

cuda_idx = 0
device = torch.device(f"cuda:{cuda_idx}")
run_ids = [3]
train_normal_agent = True
train_mango_agent = True


def run_sim(run_id, use_mango=False):
    # create the environment and the agent
    env = utils_sim.make_env(map_scale, p_frozen, one_shot, seed=run_id)
    if use_mango:
        agent = utils_sim.make_mango_agent(env, map_scale, device=device)
    else:
        agent = utils_sim.make_agent(env, map_scale, device=device)

    # train loop
    annealing_episodes, max_episodes, train_steps, episode_length = utils_sim.train_params(
        map_scale, p_frozen, one_shot
    )
    p_bar_descr = "training " + ("mango_agent" if use_mango else "normal_agent")
    randomness = np.concatenate(
        [np.linspace(1.0, 0.2, annealing_episodes), np.ones(max_episodes) * 0.2]
    )
    episode_rewards = [0.0] * 1000
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
        if float(np.mean(episode_rewards[-1000:])) >= 0.95:
            break
    return agent


if __name__ == "__main__":
    cwd = os.getcwd()
    sys.path.append(cwd)
    import utils_sim, utils_save

    for run_id in tqdm(run_ids, desc="runs", leave=False):
        dir_path = utils_save.path_to_save_dir(map_scale, p_frozen, one_shot) + "models/"
        os.makedirs(dir_path, exist_ok=True)
        run_id_str = f"run_0{run_id}" if run_id < 10 else f"run_{run_id}"

        if train_normal_agent:
            normal_agent = run_sim(run_id, use_mango=False)
            utils_save.save_to_file(
                path=dir_path + f"normal_agent_{run_id_str}.pickle", obj=normal_agent
            )

        if train_mango_agent:
            mango_agent = run_sim(run_id, use_mango=True)
            utils_save.save_to_file(
                path=dir_path + f"mango_agent_{run_id_str}.pickle", obj=mango_agent
            )
