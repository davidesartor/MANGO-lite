from typing import Any, Mapping, Protocol, Sequence, TypeVar
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ObsType = TypeVar("ObsType")


class Concept(Protocol[ObsType]):
    def abstract(self, observation: ObsType) -> npt.NDArray:
        ...


@dataclass(frozen=True, eq=False)
class IdentityConcept(Concept[npt.NDArray]):
    def abstract(self, input_state: npt.NDArray) -> npt.NDArray:
        return input_state


@dataclass
class Int2CoordConcept(Concept[int]):
    global_shape: tuple[int, int]
    cell_shape: tuple[int, int] = (1, 1)

    def abstract(self, observation: int) -> npt.NDArray:
        y, x = np.unravel_index(observation, self.global_shape)
        return np.array([y // self.cell_shape[0], x // self.cell_shape[1]])


@dataclass
class OneHotCondensation(Concept[int]):
    global_shape: tuple[int, int, int]
    name: str = "layer 1"
    condensation_window: tuple[int, int] = (2, 2)

    def abstract(self, observation: int) -> npt.NDArray:
        observation = observation.reshape(self.global_shape)
        shape_obs = observation.shape
        return observation.reshape(
            shape_obs[0] // self.condensation_window[0],
            self.condensation_window[0],
            shape_obs[1] // self.condensation_window[1],
            self.condensation_window[1],
            shape_obs[2],
        ).max(axis=(1, 3))


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
