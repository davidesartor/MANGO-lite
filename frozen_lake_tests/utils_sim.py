from typing import Any, Optional
import torch
from mango.policies.dqnet import DQNetPolicy
from mango.environments import frozen_lake
from mango.actions import grid2D
from mango import Mango, Agent


def train_params(map_scale: int, p_frozen: float | None) -> tuple[int, int, int]:
    if map_scale == 2:
        N_episodes = 1000 if p_frozen is None else 10_000
    elif map_scale == 3:
        N_episodes = 2000 if p_frozen is None else 50_000
    elif map_scale == 4:
        N_episodes = 4000 if p_frozen is None else 100_000
    else:
        raise ValueError(f"sim parameters not defined for map_scale={map_scale}")
    train_steps_per_episode = 5
    episode_length = 4**map_scale
    return N_episodes, train_steps_per_episode, episode_length


def env_params(
    map_scale: int, p_frozen: float | None = None, seed: Optional[int] = None
) -> dict[str, Any]:
    if p_frozen is None:
        return dict(map_name=f"{2**map_scale}x{2**map_scale}", seed=0)
    return dict(
        map_name="RANDOM",
        p=p_frozen,
        shape=(2**map_scale, 2**map_scale),
        goal_pos=[(0, 0), (-1, 0), (-1, -1), (0, -1)],
        seed=seed,
    )


def make_env(map_scale: int, p_frozen: float | None = None, seed: Optional[int] = None):
    params = env_params(map_scale, p_frozen, seed)
    env = frozen_lake.CustomFrozenLakeEnv(**params)
    env = frozen_lake.wrappers.ReInitOnReset(env, **params)
    env = frozen_lake.wrappers.TensorObservation(env, one_hot=True)
    return env


def abstract_actions(map_scale: int, cell_scales: list[int]):
    return [
        grid2D.SubGridMovement(
            cell_shape=(2**cell_scale, 2**cell_scale),
            grid_shape=(2**map_scale, 2**map_scale),
            agent_channel=0,
            invalid_channel=1,
        )
        for cell_scale in cell_scales
    ]


def net_params(map_scale: int, device) -> dict[str, Any]:
    map_scale = max((1, map_scale))
    repeats = 2 * map_scale - 1
    return dict(
        hidden_channels=[4 * 2**map_scale] * repeats,
        hidden_features=[],
        device=device,
    )


def policy_params(map_scale: int, lr: float, gamma: float, device) -> dict[str, Any]:
    return dict(lr=lr, gamma=gamma, net_params=net_params(map_scale, device))


def dynamic_policy_params(map_scale: int, lr: float, gamma: float, device) -> dict[str, Any]:
    return dict(policy_cls=DQNetPolicy, policy_params=policy_params(map_scale, lr, gamma, device))


def make_mango_agent(env, map_scale: int, lr: float = 3e-4, gamma=0.9, device=torch.device("cpu")):
    cell_scales = list(range(2, map_scale))
    mango_agent = Mango(
        environment=env,
        abstract_actions=abstract_actions(map_scale, cell_scales),
        policy_cls=DQNetPolicy,
        policy_params=policy_params(map_scale, lr, gamma, device),
        dynamic_policy_params=[
            dynamic_policy_params(scale, lr, gamma, device) for scale in cell_scales
        ],
    )
    mango_agent.reset()
    return mango_agent


def make_agent(
    env, map_scale: int, lr: float = 3e-4, gamma: float = 0.95, device=torch.device("cpu")
):
    agent = Agent(
        environment=env,
        policy_cls=DQNetPolicy,
        policy_params=policy_params(map_scale, lr, gamma, device),
    )
    agent.reset()
    return agent
