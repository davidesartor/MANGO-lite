from __future__ import annotations
from dataclasses import InitVar, astuple, dataclass, field, replace
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, TypeVar
from typing import Generic, NamedTuple
import spaces

ObsType = TypeVar("ObsType")
AbsObsType = TypeVar("AbsObsType")
ActType = TypeVar("ActType")
AbsActType = TypeVar("AbsActType")


@dataclass(frozen=True, slots=True)
class Transition(Generic[ObsType, ActType]):
    start_state: ObsType
    action: ActType
    next_state: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def add_abstraction(
        self, start_abstract_state: AbsObsType, next_abstract_state: AbsObsType
    ) -> Transition[tuple[ObsType, AbsObsType], ActType]:
        return replace(
            self,  # type: ignore (mypy does not understand replace changes the transition type)
            start_state=(self.start_state, start_abstract_state),
            next_state=(self.next_state, next_abstract_state),
        )

    def __iter__(self) -> Iterable[Any]:
        return iter(astuple(self))


class Concept(Protocol[ObsType, AbsObsType]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def output_space(
        self, input_space: spaces.Space[ObsType]
    ) -> spaces.Space[AbsObsType]:
        ...

    def abstract(self, observation: ObsType) -> AbsObsType:
        ...


class Policy(Protocol[ObsType, ActType]):
    def __init__(
        self,
        observation_space: spaces.Space[ObsType],
        action_space: spaces.Space[ActType],
        **kwargs,
    ):
        ...

    def set_exploration_rate(self, exploration_rate: float):
        ...

    def get_action(self, state: ObsType) -> ActType:
        ...

    def train(self, transitions: Sequence[Transition[ObsType, ActType]]):
        ...


class ActionCompatibility(Protocol[AbsActType, ObsType, ActType]):
    def __call__(
        self,
        comand: AbsActType,
        transition: AbsTransition[ObsType, AbsObsType, ActType],
    ) -> float:
        ...


class DynamicPolicy(Protocol[AbsActType, ObsType, ActType]):
    def __init__(
        self,
        command_space: spaces.Space[AbsActType],
        observation_space: spaces.Space[ObsType],
        action_space: spaces.Space[ActType],
        **kwargs,
    ):
        ...

    def set_exploration_rate(self, exploration_rate: float):
        ...

    def get_action(self, comand: AbsActType, state: ObsType) -> ActType:
        ...

    def train(
        self,
        transitions: Sequence[Transition[ObsType, ActType]],
        reward_generator: ActionCompatibility[AbsActType, ObsType, ActType],
        emphasis: Callable[[AbsActType], float] = lambda _: 1.0,
    ) -> None:
        ...


class Environment(Protocol[ObsType, ActType]):
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        ...

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict]:
        ...
