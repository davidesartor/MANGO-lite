import pickle
import torch
from mango import Mango, Agent


def path_to_save_dir(map_scale: int, p_frozen: float | None, one_shot: bool):
    path = f"frozen_lake_tests/results/{2**map_scale}x{2**map_scale}/"
    if p_frozen is None:
        return path + "1map_predefined/"
    if not one_shot:
        return path + f"1randmap_{int(p_frozen*100)}%frozen/"
    else:
        return path + f"allmaps_{int(p_frozen*100)}%frozen/"


def save_to_file(
    path: str,
    obj: Mango | Agent,
    include_env: bool = True,
    include_replay: bool = False,
    move_to_cpu: bool = True,
):
    obj.reset(options={"replay_memory": not include_replay})
    env = obj.environment
    if not include_env:
        obj.environment = None  # type: ignore
        raise Warning("Environment not saved, this may cause problems when loading")
    if move_to_cpu:
        if isinstance(obj, Mango):
            for layer in obj.abstract_layers:
                for policy in layer.policy.policies.values():  # type: ignore
                    policy.to(torch.device("cpu"))
        obj.policy.to(torch.device("cpu"))  # type: ignore
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if not include_env:
        obj.environment = env  # type: ignore


def load_from_file(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
