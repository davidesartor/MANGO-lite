from typing import Any, Mapping, Sequence, TypeVar
from dataclasses import dataclass

from protocols import Concept
from . import spaces


ObsType = TypeVar("ObsType")


class Identity(Concept[ObsType, ObsType]):
    def output_space(self, input_space: spaces.Space) -> spaces.Space:
        return input_space

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

    def output_space(self, input_space: spaces.Dict) -> spaces.Space:
        try:
            output_space: spaces.Space = input_space
            for key in self.keys:
                output_space = output_space[key]  # type: ignore
            return output_space
        except KeyError:
            raise KeyError(
                f"Incompatible space: {input_space} missing keys {self.keys}."
            )

    def abstract(self, input_state: Mapping[str, Any]) -> ObsType:
        try:
            output_state = input_state
            for key in self.keys:
                output_state = output_state[key]
            return output_state # type: ignore
        except KeyError:
            raise KeyError(
                f"Abstraction failed: {input_state} missing keys {self.keys}."
            )
