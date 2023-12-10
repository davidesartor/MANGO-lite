import pickle
from typing import Any


def path_to_save_dir(map_scale: int, p_frozen: float | None = None):
    if p_frozen is None:
        return f"results/{2**map_scale}x{2**map_scale}_PREDEFINED/"
    return f"results/{2**map_scale}x{2**map_scale}_RANDOM_p={int(p_frozen*100)}%/"


def save_to_file(
    path: str,
    obj: Any,
    include_env: bool = True,
    include_replay: bool = False,
):
    obj.reset(options={"replay_memory": not include_replay})
    env = obj.environment
    if not include_env:
        obj.environment = None
        raise Warning("Environment not saved, this may cause problems when loading")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if not include_env:
        obj.environment = env


def load_from_file(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
