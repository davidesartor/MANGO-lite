import os
import sys

# parameters for the environment
map_scale = 3
p_frozen = 0.8

if __name__ == "__main__":
    cwd = os.getcwd()
    sys.path.append(cwd)
    import utils_save

    dir_path = utils_save.path_to_save_dir(map_scale, p_frozen)
    for filename in os.listdir(dir_path + "models/"):
        normal_agent = utils_save.load_from_file(dir_path + "models/" + filename)
        utils_save.save_to_file(path=dir_path + "models/" + filename, obj=normal_agent)
        agent = utils_save.load_from_file(dir_path + "models/" + filename)
        print(agent.policy.device)
