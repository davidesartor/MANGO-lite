from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Generic, Literal, TypeVar, Protocol

import numpy as np
from utils.buffers import obs_not_equal
from utils.spaces import CompositeSpace, FiniteSpace, MultiFiniteSpace, SpaceType
from utils.concepts import ConceptFunction

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class CompatibilityFunction(Protocol[ActType, ObsType]):
    action_space: SpaceType[ActType]
    observation_space: SpaceType[ObsType]
    default_val: float = 0.0

    def __call__(
        self, transition: tuple[ObsType, ObsType] | None
    ) -> Callable[[ActType], float]:
        if transition is None:
            return lambda _: self.default_val
        return self.compatibility_function(*transition)

    def compatibility_function(
        self, start_state: ObsType, end_state: ObsType
    ) -> Callable[[ActType], float]:
        ...


@dataclass(frozen=True, eq=False)
class FullCompatibility(CompatibilityFunction[ActType, ObsType]):
    action_space: SpaceType[ActType]
    observation_space: SpaceType[ObsType]
    default_val: float = 0.0

    def compatibility_function(
        self, start_state: ObsType, end_state: ObsType
    ) -> Callable[[ActType], float]:
        if obs_not_equal(start_state, end_state):
            return lambda action: 1.0
        return lambda action: -1.0



# region Minigrif compatibility functions


@dataclass(frozen=True, eq=False)
class SingleConceptCompatibilityAdapter(
    CompatibilityFunction[ActType, dict[str, Any]], Generic[ActType, ObsType]
):
    concept: InitVar[ConceptFunction[Any, ObsType]]
    compatibility: CompatibilityFunction[ActType, ObsType]
    default_val: float = 0.0
    observation_space: CompositeSpace = field(init=False)
    action_space: SpaceType[ActType] = field(init=False)
    concept_name: str = field(init=False)
    

    def __post_init__(self, concept):
        outer_observation_space = CompositeSpace(
            {concept.name: concept.output_observation_space}
        )
        object.__setattr__(self, "action_space", self.compatibility.action_space)
        object.__setattr__(self, "observation_space", outer_observation_space)
        object.__setattr__(self, "concept_name", concept.name)

    def __call__(
        self, transition: tuple[dict[str, Any], dict[str, Any]] | None
    ) -> Callable[[ActType], float]:
        if transition is None:
            return super().__call__(transition)
        else:
            return self.compatibility(tuple(s[self.concept_name] for s in transition))

    def compatibility_function(
        self, start_state: dict[str, Any], end_state: dict[str, Any]
    ) -> Callable[[ActType], float]:
        return self.compatibility(
            (start_state[self.concept_name], end_state[self.concept_name])
        )


@dataclass(frozen=True, eq=False)
class MultiConceptCompatibilityAdapter(CompatibilityFunction[ActType, dict[str, Any]]):
    compatibilities: dict[Any, CompatibilityFunction[ActType, Any]]
    default_val: float = 0.0
    observation_space: CompositeSpace = field(init=False)
    action_space: MultiFiniteSpace = field(init=False)

    def __post_init__(self):
        outer_observation_space = CompositeSpace(
            {
                key: compatibility.observation_space
                for key, compatibility in self.compatibilities.items()
            }
        )
        outer_action_space = MultiFiniteSpace(
            {
                key: compatibility.action_space  # type: ignore
                for key, compatibility in self.compatibilities.items()
            }
        )
        object.__setattr__(self, "action_space", outer_action_space)
        object.__setattr__(self, "observation_space", outer_observation_space)

    
    def compatibility_function(
        self, start_state: dict[Any, Any], end_state: dict[Any, Any]
    ) -> Callable[[ActType], float]:
        def compatibility(action):
            key = self.action_space.reverse_map[action]
            return self.compatibilities[key]((start_state[key], end_state[key]))(action)
        return compatibility


@dataclass(frozen=True, eq=False)
class GridAdjacencyCompatibility(CompatibilityFunction[Any, int]):
    observation_space: FiniteSpace[int]
    grid_shape: tuple[int, int]
    action_space = FiniteSpace(["U", "D", "L", "R"])
    default_val: float = 0.0

    def compatibility_function(
        self, start_state: int, end_state: int
    ) -> Callable[[Literal["U", "D", "L", "R"]], float]:
        y1, x1 = np.unravel_index(start_state, self.grid_shape)
        y2, x2 = np.unravel_index(end_state, self.grid_shape)
        deltayx = y2 - y1, x2 - x1

        def compatibility(action):
            if action not in self.action_space:
                raise ValueError(f"Invalid action {action}")
            target_deltayx = {
                "U": (1, 0),
                "D": (-1, 0),
                "L": (0, -1),
                "R": (0, 1),
            }[action]
            return 2 * float(deltayx == target_deltayx) - 1

        return compatibility


# endregion

# region future compatibility functions


class AdjacencyFunction(Protocol[ObsType, ActType]):
    action_space: SpaceType[ActType]
    observation_space: SpaceType[ObsType]

    def __call__(
        self, action: ActType, transition: tuple[ObsType, ObsType] | None
    ) -> float | None:
        if transition is None:
            return None
        return self.transition_probability(action, *transition)

    def transition_probability(
        self, action: ActType, start_state: ObsType, end_state: ObsType
    ) -> float:
        ...


@dataclass(frozen=True, eq=False)
class DecompositionAdjacencyFunction(AdjacencyFunction[ObsType, ActType]):
    adj_components: dict[ActType, Callable[[ObsType, ObsType], float]]
    action_space: FiniteSpace[ActType]
    observation_space: SpaceType[ObsType]

    def transition_probability(
        self, action: ActType, start_state: int, end_state: int
    ) -> float:
        return self.adj_components[action](start_state, end_state)


@dataclass(frozen=True, eq=False)
class AdjacencyCompatibility(CompatibilityFunction[ActType, ObsType]):
    adj_function: AdjacencyFunction[ObsType, ActType]
    prob2reward: Callable[[float], float] = field(default=lambda x: 2 * x - 1)

    @property
    def action_space(self) -> SpaceType[ActType]:
        return self.adj_function.action_space

    @property
    def observation_space(self) -> SpaceType[ObsType]:
        return self.adj_function.observation_space

    def compatibility_function(
        self, start_state: ObsType, end_state: ObsType
    ) -> Callable[[ActType], float]:
        return lambda action: self.prob2reward(
            self.adj_function.transition_probability(action, start_state, end_state)
        )


# endregion
