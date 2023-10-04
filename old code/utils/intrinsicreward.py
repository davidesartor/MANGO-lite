from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
from utils.buffers import ObsBuffer
from utils.compatibilityfunctions import CompatibilityFunction
from utils.spaces import SASRI

UpObsType = TypeVar("UpObsType")
UpActType = TypeVar("UpActType", contravariant=True)
LowObsType = TypeVar("LowObsType")
LowActType = TypeVar("LowActType")


class IntrinsicRewardGenerator(Protocol[UpActType, LowObsType, LowActType]):
    def generate_info(
        self, sasri: SASRI[LowObsType, LowActType]
    ) -> SASRI[LowObsType, LowActType]:
        ...

    def generate_reward(
        self, comand: UpActType, sasri: SASRI[LowObsType, LowActType]
    ) -> SASRI[LowObsType, LowActType]:
        ...


@dataclass(frozen=True, eq=True)
class StateTransitionRewardGenerator(
    IntrinsicRewardGenerator[UpActType, LowObsType, LowActType],
    Generic[UpObsType, UpActType, LowObsType, LowActType],
):
    buffer: ObsBuffer[UpObsType]
    compatibility: CompatibilityFunction[UpActType, UpObsType]

    def generate_info(
        self, sasri: SASRI[LowObsType, LowActType]
    ) -> SASRI[LowObsType, LowActType]:
        sasri.info["buffertransition"] = self.buffer.current_transition
        return sasri

    def generate_reward(
        self, comand: UpActType, sasri: SASRI[LowObsType, LowActType]
    ) -> SASRI[LowObsType, LowActType]:
        intrinsic_reward = self.compatibility(sasri.info["buffertransition"])(comand)
        return SASRI.from_sasri_template(template=sasri, reward=intrinsic_reward)
