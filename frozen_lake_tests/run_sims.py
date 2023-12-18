import os
import sys
from tqdm import tqdm
import numpy as np

# parameters for the environment
map_scale = 2
p_frozen = 0.8
run_ids = [0, 1]
train_normal_agent = True
train_mango_agent = True


def run_sim(run_id, use_mango=False):
    # create the environment and the agent
    env = utils_sim.make_env(map_scale, p_frozen, seed=run_id)
    if use_mango:
        agent = utils_sim.make_mango_agent(env, map_scale)
    else:
        agent = utils_sim.make_agent(env, map_scale)

    # train loop
    N_episodes, train_steps_per_episode, episode_length = utils_sim.train_params(
        map_scale, p_frozen
    )
    p_bar_descr = "training " + ("mango_agent" if use_mango else "normal_agent")
    for episode_idx, randomness in enumerate(
        tqdm(np.linspace(1.0, 0.0, N_episodes), desc=p_bar_descr, leave=False)
    ):
        randomness = 0.0 if episode_idx % 2 else randomness
        agent.run_episode(randomness, episode_length)
        for _ in range(train_steps_per_episode):
            agent.train()
    return agent


if __name__ == "__main__":
    cwd = os.getcwd()
    sys.path.append(cwd)
    import utils_sim, utils_save

    for run_id in tqdm(run_ids, desc="runs"):
        dir_path = utils_save.path_to_save_dir(map_scale, p_frozen) + "models/"
        os.makedirs(dir_path, exist_ok=True)

        if train_normal_agent:
            normal_agent = run_sim(run_id, use_mango=False)
            utils_save.save_to_file(
                path=dir_path + f"normal_agent_run_{run_id}.pickle", obj=normal_agent
            )

        if train_mango_agent:
            mango_agent = run_sim(run_id, use_mango=True)
            utils_save.save_to_file(
                path=dir_path + f"mango_agent_run_{run_id}.pickle", obj=mango_agent
            )
