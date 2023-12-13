import sys

sys.path.append("..")
from typing import Any
import torch
from mango.policies.dqnet import DQNetPolicy
from mango.environments import frozen_lake
from mango.actions import grid2D
from mango import Mango, Agent


def env_params(map_scale: int, p_frozen: float | None = None):
    if p_frozen is None:
        return dict(map_name=f"{2**map_scale}x{2**map_scale}")
    return dict(
        map_name="RANDOM",
        p=p_frozen,
        shape=(2**map_scale, 2**map_scale),
        goal_pos=[(0, 0), (-1, 0), (-1, -1), (0, -1)],
        # start_pos=[(0, 0), (-1, 0), (-1, -1), (0, -1)],
    )


def make_env(map_scale: int, p_frozen: float | None = None):
    env = frozen_lake.CustomFrozenLakeEnv(**env_params(map_scale, p_frozen))  # type: ignore
    env = frozen_lake.wrappers.ReInitOnReset(env, **env_params(map_scale, p_frozen))
    env = frozen_lake.wrappers.TensorObservation(env, one_hot=True)
    return env


def abstract_actions(map_scale: int, cell_scales: list[int], gamma: float):
    return [
        grid2D.SubGridMovement(
            cell_shape=(2**cell_scale, 2**cell_scale),
            grid_shape=(2**map_scale, 2**map_scale),
            agent_channel=0,
            invalid_channel=1,
            success_reward=1.0,
            failure_reward=-1.0,
        )
        for cell_scale in cell_scales
    ]


def net_params(map_scale: int):
    map_scale = max((1, map_scale))
    repeats = 2 * map_scale - 1
    return dict(
        hidden_channels=[4 * 2**map_scale] * repeats,
        hidden_features=[],
        device=torch.device("cuda") if torch.cuda.is_available() else None,
    )


def dynamic_policy_params(map_scale: int, lr: float, gamma: float) -> dict[str, Any]:
    return dict(
        policy_cls=DQNetPolicy,
        policy_params=dict(lr=lr, gamma=gamma, net_params=net_params(map_scale)),
    )


def make_mango_agent(
    env, map_scale: int, lr: float = 3e-4, gamma: float = 0.95, gamma_options: float = 0.95
):
    cell_scales = list(range(1, map_scale - 1))
    mango_agent = Mango(
        environment=env,
        abstract_actions=abstract_actions(map_scale, cell_scales, gamma_options),
        policy_cls=DQNetPolicy,
        policy_params=dict(lr=lr, gamma=gamma, net_params=net_params(map_scale)),
        dynamic_policy_params=[
            dynamic_policy_params(scale, lr, gamma_options) for scale in cell_scales
        ],
    )
    mango_agent.reset()
    return mango_agent


def make_agent(env, map_scale: int, lr: float = 3e-4, gamma: float = 0.95):
    agent = Agent(
        environment=env,
        **dynamic_policy_params(map_scale, lr, gamma),
    )
    agent.reset()
    return agent
