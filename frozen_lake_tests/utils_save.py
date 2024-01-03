import pickle
from typing import Any
import numpy as np
import torch
from mango import Mango, Agent


def path_to_save_dir(map_base, map_scale: int, p_frozen: float | None, one_shot: bool):
    path = f"frozen_lake_tests/results/{map_base**map_scale}x{map_base**map_scale}/"
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
    convert_logs_to_numpy: bool = True,
    move_to_cpu: bool = True,
):
    obj.reset(options={"replay_memory": not include_replay})
    env = obj.environment
    if not include_env:
        obj.environment = None
        raise Warning("Environment not saved, this may cause problems when loading")
    if move_to_cpu:
        if isinstance(obj, Mango):
            for layer in obj.abstract_layers:
                for policy in layer.policy.policies.values():
                    policy.to(torch.device("cpu"))
        obj.policy.to(torch.device("cpu"))

    if convert_logs_to_numpy:
        if isinstance(obj, Mango):
            for layer in obj.abstract_layers:
                layer.intrinsic_reward_log = tuple(
                    np.array(log) for log in layer.intrinsic_reward_log
                )
                layer.train_loss_log = tuple(np.array(log) for log in layer.train_loss_log)
                layer.episode_length_log = np.array(obj.episode_length_log)
        obj.reward_log = np.array(obj.reward_log)
        obj.train_loss_log = np.array(obj.train_loss_log)
        obj.episode_length_log = np.array(obj.episode_length_log)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if not include_env:
        obj.environment = env


def load_from_file(path: str, device=None) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, Mango):
        for layer in obj.abstract_layers:
            for policy in layer.policy.policies.values():
                policy.to(device)
    obj.policy.to(device)

    if isinstance(obj, Mango):
        for layer in obj.abstract_layers:
            layer.intrinsic_reward_log = tuple(list(log) for log in layer.intrinsic_reward_log)
            layer.train_loss_log = tuple(list(log) for log in layer.train_loss_log)
            layer.episode_length_log = list(obj.episode_length_log)
    obj.reward_log = list(obj.reward_log)
    obj.train_loss_log = list(obj.train_loss_log)
    obj.episode_length_log = list(obj.episode_length_log)

    return obj
