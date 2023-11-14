from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, Iterator, Optional, Sequence, SupportsFloat, TypeVar
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from .actions import ActionCompatibility
from .concepts import Concept, IdentityConcept
from .masking_window import Window
from .dynamicpolicies import DQnetPolicyMapper, DynamicPolicy
from .utils import ReplayMemory, Transition, torch_style_repr


ObsType = TypeVar("ObsType")

actions_name = ["left", "down", "right", "up"]

@dataclass(eq=False, slots=True, repr=False)
class MangoEnv(Generic[ObsType]):
    concept: Concept[ObsType]
    environment: gym.Env[ObsType, int]
    abs_state: npt.NDArray = field(init=False)
    masked_state: npt.NDArray = field(init=False)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.environment.action_space  # type: ignore

    def step(
        self, action: int, randomness: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        if np.random.rand() < randomness:
            action = int(self.action_space.sample())
        env_state, reward, term, trunc, info = self.environment.step(action)
        self.abs_state = self.concept.abstract(env_state)
        self.masked_state = env_state
        info["mango:trajectory"] = [env_state]
        return env_state, reward, term, trunc, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        env_state, _, _, _, info = self.environment.reset(seed=seed, options=options)
        self.abs_state = self.concept.abstract(env_state)
        self.masked_state = env_state
        return env_state, info

    def __repr__(self) -> str:
        return torch_style_repr(
            self.__class__.__name__,
            {"concept": str(self.concept), "environment": str(self.environment)},
        )


@dataclass(eq=False, slots=True, repr=False)
class MangoLayer(Generic[ObsType]):
    concept: Concept[ObsType]
    mask: Window[ObsType]
    action_compatibility: ActionCompatibility
    lower_layer: MangoLayer[ObsType] | MangoEnv[ObsType]
    max_steps: int = 10
    verbose: int = 0
    replay_memory: ReplayMemory[tuple[Transition, npt.NDArray, npt.NDArray]] = field(
        init=False, default_factory=ReplayMemory
    )
    policy: DynamicPolicy = field(init=False)
    abs_state: npt.NDArray = field(init=False)
    masked_state: npt.NDArray = field(init=False)
    # record the intrinsic rewards
    debug_log: tuple[list, ...] = field(default_factory=lambda: ([], [], [], []))

    def __post_init__(self) -> None:
        self.policy = DQnetPolicyMapper(
            comand_space=self.action_compatibility.action_space,
            action_space=self.lower_layer.action_space,
        )

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.action_compatibility.action_space

    def step(
        self, action: int, randomness: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        abs_states_low = [self.lower_layer.abs_state]
        abs_states_up = [self.abs_state]
        states_low = [self.lower_layer.masked_state]
        states_up = [self.masked_state]
        env_state_trajectory: list[ObsType] = []
        transitions_low: list[Transition] = []
        if self.verbose>=1:
            print(f"-"*(2-int(self.concept.name))+f"Layer {self.concept.name} action {actions_name[action]}")
        for step in range(max((self.max_steps, 1))):
            low_action = self.policy.get_action(
                comand=action, state=states_low[-1], randomness=randomness
            )
            env_state, reward, term, trunc, info = self.lower_layer.step(low_action, randomness=0.0)
            
            env_state_trajectory.extend(info["mango:trajectory"])
            self.abs_state = self.concept.abstract(env_state)
            self.masked_state = self.mask.abstract(env_state,self.abs_state)

            abs_states_low.append(self.lower_layer.abs_state)
            abs_states_up.append(self.abs_state)
            states_low.append(self.lower_layer.masked_state)
            states_up.append(self.masked_state)
            transition_low = Transition(
                states_low[-2], low_action, states_low[-1], reward, term, trunc, info
            )
            transitions_low.append(transition_low)

            if term or trunc or not np.all(abs_states_up[-1] == abs_states_up[-2]):
                break

        self.replay_memory.extend(zip(transitions_low, abs_states_up[:-1], abs_states_up[1:]))
        accumulated_reward = sum(float(t_low.reward) for t_low in transitions_low)
        if self.verbose >= 2:
            print(f"-"*(2-int(self.concept.name))+f"Layer {self.concept.name} reward {accumulated_reward}")
        infos = {k: v for t_low in transitions_low for k, v in t_low.info.items()}
        infos["mango:trajectory"] = env_state_trajectory

        # for debug log the intrinsic reward
        self.debug_log[action].append(
            self.action_compatibility(action, abs_states_up[-1], abs_states_up[-2])
        )
        return env_state, accumulated_reward, term, trunc, infos  # type: ignore

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        env_state, info = self.lower_layer.reset(seed=seed, options=options)
        self.abs_state = self.concept.abstract(env_state)
        self.masked_state = self.mask.abstract(env_state,self.abs_state)
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
        masks: Sequence[Window[ObsType]],
        base_concept: Concept[ObsType] = IdentityConcept(),
        verbose: int = 0,
    ) -> None:
        self.environment = MangoEnv(base_concept, environment)
        self.abstract_layers: list[MangoLayer[ObsType]] = []
        for concept, mask, compatibility in zip(concepts, masks, action_compatibilities):
            self.abstract_layers.append(
                MangoLayer(concept, mask, compatibility, self.layers[-1],verbose = verbose)
            )
        self.reset()

    @property
    def layers(self) -> tuple[MangoEnv[ObsType] | MangoLayer[ObsType], ...]:
        return (self.environment, *self.abstract_layers)
    

    @property
    def option_space(self) -> tuple[gym.spaces.Discrete, ...]:
        return tuple(layer.action_space for layer in self.layers)
    
    def change_verbose(self, verbose: int) -> None:
        self.verbose = verbose
        for layer in self.abstract_layers:
            layer.verbose = verbose

    def execute_option(
        self, action: int, layer_idx: int, randomness: float = 0.0
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        return self.layers[layer_idx].step(action, randomness)

    def train(self, layer_idx: int) -> None:
        if layer_idx == 0:
            raise ValueError("Cannot train base layer")
        layer = self.abstract_layers[layer_idx - 1]
        if layer.replay_memory.size > 1:
            layer.policy.train(
                layer.replay_memory.sample(),
                layer.action_compatibility,
            )

    def explore(
        self, layer_idx: int, randomness: float, max_steps: int = 5
    ) -> tuple[ObsType, float, bool, bool, dict]:
        env_state, info = self.reset()
        accumulated_reward, term, trunc = 0.0, False, False
        for _ in range(max_steps):
            env_state, reward, term, trunc, info = self.execute_option(
                action=int(self.option_space[layer_idx].sample()),
                layer_idx=layer_idx,
                randomness=randomness,
            )
            accumulated_reward += float(reward)
            if term or trunc:
                break
        return env_state, accumulated_reward, term, trunc, info

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

