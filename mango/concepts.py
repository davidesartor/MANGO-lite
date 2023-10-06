from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar
from dataclasses import dataclass

from gymnasium import spaces

ObsType = TypeVar("ObsType")
AbsObsType = TypeVar("AbsObsType")
AbsActType = TypeVar("AbsActType")


class Concept(Protocol[ObsType, AbsObsType]):
    def abstract(self, observation: ObsType) -> AbsObsType:
        ...


class ExtendedConcept(Protocol[ObsType, AbsObsType, AbsActType]):
    @property
    def comand_space(self) -> spaces.Space[AbsActType]:
        ...

    def compatibility(
        self, comand: AbsActType, start_state: AbsObsType, next_state: AbsObsType
    ) -> float:
        ...

    def abstract(self, observation: ObsType) -> AbsObsType:
        ...


class IdentityConcept(Concept[Any, Any]):
    def abstract(self, input_state: Any) -> Any:
        return input_state


@dataclass(frozen=True, eq=False)
class IdentityExtendedConcept(Generic[AbsActType]):
    comand_space: spaces.Space[AbsActType]

    def compatibility(
        self, comand: AbsActType, start_state: Any, next_state: Any
    ) -> float:
        return 1.0

    def abstract(self, input_state: Any) -> Any:
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
