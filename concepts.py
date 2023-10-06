from typing import Any, Mapping, Protocol, Sequence, TypeVar
from dataclasses import dataclass

from gymnasium import spaces

ObsType = TypeVar("ObsType")
AbsObsType = TypeVar("AbsObsType")
AbsActType = TypeVar("AbsActType")


class Concept(Protocol[ObsType, AbsObsType]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def abstract(self, observation: ObsType) -> AbsObsType:
        ...


class ExtendedConcept(Protocol[ObsType, AbsObsType, AbsActType]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def comand_space(self) -> spaces.Space[AbsActType]:
        ...

    def abstract(self, observation: ObsType) -> AbsObsType:
        ...

    def compatibility(
        self, comand: AbsActType, start_state: AbsObsType, next_state: AbsObsType
    ) -> float:
        ...


class Identity(Concept[ObsType, ObsType]):
    def abstract(self, input_state: ObsType) -> ObsType:
        return input_state


@dataclass(frozen=True, eq=False)
class Strip(Concept[Mapping[str, Any], ObsType]):
    key: str | Sequence[str]

    @property
    def keys(self) -> Sequence[str]:
        return [self.key] if isinstance(self.key, str) else self.key

    @property
    def name(self) -> str:
        return f"Strip[" + "][ ".join([str(k) for k in self.keys]) + "]Concept"

    def abstract(self, input_state: Mapping[str, Any]) -> ObsType:
        try:
            output_state = input_state
            for key in self.keys:
                output_state = output_state[key]
            return output_state  # type: ignore
        except KeyError:
            raise KeyError(
                f"Abstraction failed: {input_state} missing keys {self.keys}."
            )
