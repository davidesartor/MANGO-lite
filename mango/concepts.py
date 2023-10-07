from typing import Any, Mapping, Protocol, Sequence, TypeVar
from dataclasses import dataclass

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

ObsType = TypeVar("ObsType")


class Concept(Protocol[ObsType]):
    def abstract(self, observation: ObsType) -> npt.NDArray:
        ...


class ActionCompatibility(Protocol):
    action_space: spaces.Discrete

    def __call__(
        self, comand: int, start_state: npt.NDArray, next_state: npt.NDArray
    ) -> float:
        ...


@dataclass(frozen=True, eq=False)
class IdentityConcept(Concept[npt.NDArray]):
    def abstract(self, input_state: npt.NDArray) -> npt.NDArray:
        return input_state


@dataclass(frozen=True, eq=False)
class FullCompatibility(ActionCompatibility):
    action_space: spaces.Discrete

    def __call__(self, comand: Any, start_state: Any, next_state: Any) -> float:
        return 1.0


@dataclass(frozen=True, eq=False)
class Strip(Concept[Mapping[str, Any]]):
    key: str | Sequence[str]

    @property
    def keys(self) -> Sequence[str]:
        return [self.key] if isinstance(self.key, str) else self.key

    @property
    def name(self) -> str:
        return f"Strip[" + "][ ".join([str(k) for k in self.keys]) + "]Concept"

    def abstract(self, input_state: Mapping[str, Any]) -> npt.NDArray:
        try:
            output_state = input_state
            for key in self.keys:
                output_state = output_state[key]
            return np.array(output_state)
        except KeyError:
            raise KeyError(
                f"Abstraction failed: {input_state} missing keys {self.keys}."
            )
