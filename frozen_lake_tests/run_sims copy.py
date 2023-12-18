import os
import sys
import torch
from tqdm import tqdm
import numpy as np

# parameters for the environment
map_scale = 2
p_frozen = None
device = torch.device("cpu")
run_ids = [0, 1, 2]
train_normal_agent = True
train_mango_agent = True


if __name__ == "__main__":
    cwd = os.getcwd()
    sys.path.append(cwd)
    import utils_save

    for run_id in tqdm(run_ids, desc="runs"):
        # load normal agent models one by one
        dir_path = utils_save.path_to_save_dir(map_scale, p_frozen)
        for filename in os.listdir(dir_path + "models/"):
            if not filename.startswith("mango_agent"):
                continue
            normal_agent = utils_save.load_from_file(dir_path + "models/" + filename)
            utils_save.save_to_file(
                path=dir_path + "models/" + f"normal_agent_run_{run_id}.pickle", obj=normal_agent
            )
            agent = utils_save.load_from_file(dir_path + "models/" + filename)
            print(agent.policy.device)

        # load mango agent models one by one
        dir_path = utils_save.path_to_save_dir(map_scale, p_frozen)
        for filename in os.listdir(dir_path + "models/"):
            if not filename.startswith("mango_agent"):
                continue
            mango_agent = utils_save.load_from_file(dir_path + "models/" + filename)
            utils_save.save_to_file(
                path=dir_path + f"mango_agent_run_{run_id}.pickle", obj=mango_agent
            )
