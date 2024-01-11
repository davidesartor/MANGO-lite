from typing import Any, Optional
import torch
from mango.policies.dqnet import DQNetPolicy
from mango.environments import frozen_lake
from mango.actions import grid2D
from mango import Mango, Agent


def train_params(
    map_base: int, map_scale: int, p_frozen: float | None, one_shot
) -> tuple[int, int, int, int]:
    max_episodes = 10 ** (map_scale + map_base)
    if not one_shot or p_frozen is None:
        max_episodes = max_episodes // 2
    annealing_episodes = max_episodes // 10
    max_episodes = max_episodes - annealing_episodes
    train_steps_per_episode = 5
    episode_length = (map_base**map_scale) ** 2
    return annealing_episodes, max_episodes, train_steps_per_episode, episode_length


def env_params(
    map_base: int, map_scale: int, p_frozen: float | None, seed: Optional[int] = None
) -> dict[str, Any]:
    if p_frozen is None:
        return dict(map_name=f"{map_base**map_scale}x{map_base**map_scale}", seed=0)
    return dict(
        map_name="RANDOM",
        p=p_frozen,
        shape=(map_base**map_scale, map_base**map_scale),
        start_pos=[(0, 0)],
        goal_pos=[(-1, -1)],
        seed=seed,
    )


def make_env(
    map_base: int, map_scale: int, p_frozen: float | None, one_shot, seed: Optional[int] = None
):
    params = env_params(map_base, map_scale, p_frozen, seed)
    env = frozen_lake.CustomFrozenLakeEnv(**params)
    if one_shot:
        env = frozen_lake.wrappers.ReInitOnReset(env, **params)
    env = frozen_lake.wrappers.TensorObservation(env, one_hot=True)
    return env


def abstract_actions(map_base: int, map_scale: int, cell_scales: list[int], mask_state: bool):
    return [
        grid2D.SubGridMovement(
            cell_shape=(map_base**cell_scale, map_base**cell_scale),
            grid_shape=(map_base**map_scale, map_base**map_scale),
            agent_channel=0,
            invalid_channel=1,
            mask_state=mask_state,
        )
        for cell_scale in cell_scales
    ]


def net_params(map_base: int, map_scale: int, device) -> dict[str, Any]:
    map_scale = max((1, map_scale))
    repeats = map_base * map_scale - 1
    return dict(
        hidden_channels=[4 * 2**map_scale] * repeats,
        hidden_features=[],
        device=device,
    )


def policy_params(map_base: int, map_scale: int, lr: float, gamma: float, device) -> dict[str, Any]:
    return dict(lr=lr, gamma=gamma, net_params=net_params(map_base, map_scale, device))


def dynamic_policy_params(
    map_base: int, map_scale: int, lr: float, gamma: float, device
) -> dict[str, Any]:
    return dict(
        policy_cls=DQNetPolicy, policy_params=policy_params(map_base, map_scale, lr, gamma, device)
    )


def make_mango_agent(
    env,
    map_base: int,
    map_scale: int,
    mask_state: bool = True,
    lr: float = 3e-4,
    gamma=0.95,
    device=torch.device("cpu"),
):
    cell_scales = list(range(1, map_scale))
    mango_agent = Mango(
        environment=env,
        abstract_actions=abstract_actions(map_base, map_scale, cell_scales, mask_state),
        policy_cls=DQNetPolicy,
        policy_params=policy_params(map_base, map_scale, lr, gamma, device),
        dynamic_policy_params=[
            dynamic_policy_params(map_base, scale, lr, gamma, device) for scale in cell_scales
        ],
    )
    return mango_agent


def make_agent(
    env,
    map_base: int,
    map_scale: int,
    lr: float = 3e-4,
    gamma: float = 0.95,
    device=torch.device("cpu"),
):
    agent = Agent(
        environment=env,
        policy_cls=DQNetPolicy,
        policy_params=policy_params(map_base, map_scale, lr, gamma, device),
    )
    return agent
