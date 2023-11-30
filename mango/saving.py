import pickle
from .mango import Mango
from .agents import Agent


def save_to(
    path: str,
    obj: Mango | Agent,
    include_env: bool = True,
    include_replay: bool = False,
):
    obj.reset(options={"replay_memory": not include_replay})
    env = obj.environment
    if not include_env:
        obj.environment = None  # type: ignore
        raise Warning("Environment not saved, this may cause problems when loading")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if not include_env:
        obj.environment = env  # type: ignore


def load_from(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
