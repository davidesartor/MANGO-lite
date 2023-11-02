from __future__ import annotations
from dataclasses import dataclass, field
from math import e
import re
from typing import Generic, Optional, Sequence, SupportsFloat, TypeVar
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .actions import ActionCompatibility
from .concepts import Concept, IdentityConcept
from .dynamicpolicies import DQnetPolicyMapper
from .utils import ReplayMemory, Transition, torch_style_repr


ObsType = TypeVar("ObsType")


@dataclass(eq=False, slots=True, repr=False)
class MangoEnv(Generic[ObsType]):
    concept: Concept[ObsType]
    environment: gym.Env[ObsType, int]
    abs_state: npt.NDArray = field(init=False)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.environment.action_space  # type: ignore

    def step(
        self, action: int, epsilon: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        if np.random.rand() < epsilon:
            action = int(self.action_space.sample())
        env_state, reward, term, trunc, info = self.environment.step(action)
        self.abs_state = self.concept.abstract(env_state)
        info["mango:trajectory"] = [env_state]
        return env_state, reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        env_state, info = self.environment.reset(seed=seed, options=options)
        self.abs_state = self.concept.abstract(env_state)
        return env_state, info

    def __repr__(self) -> str:
        return torch_style_repr(
            self.__class__.__name__,
            {"concept": str(self.concept), "environment": str(self.environment)},
        )


@dataclass(eq=False, slots=True, repr=False)
class MangoLayer(Generic[ObsType]):
    concept: Concept[ObsType]
    action_compatibility: ActionCompatibility
    lower_layer: MangoLayer[ObsType] | MangoEnv[ObsType]
    max_steps: int = 5

    replay_memory: ReplayMemory = field(init=False)
    policy: DQnetPolicyMapper = field(init=False)
    abs_state: npt.NDArray = field(init=False)
    debug_log: tuple[list, ...] = field(default_factory=lambda: ([], [], [], []))

    def __post_init__(self) -> None:
        self.replay_memory = ReplayMemory()
        self.policy = DQnetPolicyMapper(
            comand_space=self.action_compatibility.action_space,
            action_space=self.lower_layer.action_space,
        )

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.action_compatibility.action_space

    def step(
        self, action: int, epsilon: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        transitions = []
        env_state_trajectory = []
        low_state = self.lower_layer.abs_state
        up_state = self.abs_state

        for step in range(max(self.max_steps, 1)):
            low_action = self.policy.get_action(comand=action, state=low_state)
            if np.random.rand() < epsilon:
                low_action = int(self.action_space.sample())
                
            env_state, reward, term, trunc, info = self.lower_layer.step(low_action)
            env_state_trajectory.extend(info["mango:trajectory"])
            self.abs_state = self.concept.abstract(env_state)
            low_next_state, up_next_state = self.lower_layer.abs_state, self.abs_state

            low_term = term or not np.all(up_state == up_next_state)
            low_transition = Transition(
                low_state, low_action, low_next_state, reward, low_term, trunc, info
            )
            up_transition = Transition(
                up_state, action, up_next_state, reward, term, trunc, info
            )
            transitions.append((low_transition, up_transition))
            
            if low_term or trunc:
                break

            low_state, up_state = low_next_state, up_next_state

        for t in transitions:
            self.replay_memory.push(t)

        accumulated_reward = sum([t_low.reward for t_low, t_up in transitions])
        infos = {k: v for t_low, t_up in transitions for k, v in t_low.info.items()}
        infos["mango:trajectory"] = env_state_trajectory

        self.debug_log[action].append(
            self.action_compatibility(action, up_state, up_next_state)  # type: ignore
        )
        return env_state, accumulated_reward, term, trunc, infos  # type: ignore

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        env_state, info = self.lower_layer.reset(seed=seed, options=options)
        self.abs_state = self.concept.abstract(env_state)
        return env_state, info

    def __repr__(self) -> str:
        params = {
            "concept": str(self.concept),
            "act_comp": str(self.action_compatibility),
            "policy": str(self.policy),
        }
        return torch_style_repr(self.__class__.__name__, params)


class Mango(Generic[ObsType]):
    def __init__(
        self,
        environment: gym.Env[ObsType, int],
        concepts: Sequence[Concept[ObsType]],
        action_compatibilities: Sequence[ActionCompatibility],
        base_concept: Concept[ObsType] = IdentityConcept(),
    ) -> None:
        self.environment = MangoEnv(base_concept, environment)
        self.layers: list[MangoEnv | MangoLayer] = [self.environment]
        for concept, compatibility in zip(concepts, action_compatibilities):
            self.layers.append(MangoLayer(concept, compatibility, self.layers[-1]))
        self.reset()

    @property
    def option_space(self) -> tuple[gym.spaces.Discrete, ...]:
        return tuple(layer.action_space for layer in [self.environment, *self.layers])
    
    @property
    def abstract_layers(self) -> tuple[MangoLayer, ...]:
        return tuple(self.layers[1:]) # type: ignore

    def execute_option(
        self, action: int, layer_idx: int, epsilon: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        return self.layers[layer_idx].step(action, epsilon)

    def train(self, layer_idx: int) -> None:
        if layer_idx == 0:
            raise ValueError("Cannot train base layer")
        layer: MangoLayer = self.layers[layer_idx]  # type: ignore
        if layer.replay_memory.size > 1:
            layer.policy.train(
                layer.replay_memory.sample(),
                layer.action_compatibility,
            )

    def explore(self, layer_idx: int, epsilon: float, max_steps: int = 1) -> None:
        if layer_idx == 0:
            raise ValueError("Cannot train base layer")
        self.reset()
        for _ in range(max_steps):
            self.execute_option(
                action=int(self.option_space[layer_idx].sample()),
                layer_idx=layer_idx,
                epsilon=epsilon,
            )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        return self.layers[-1].reset(seed=seed, options=options)

    def __repr__(self) -> str:
        params = {
            "0": str(self.environment),
            **{f"{i+1}": str(layer) for i, layer in enumerate(self.layers)},
        }
        return torch_style_repr(self.__class__.__name__, params)
