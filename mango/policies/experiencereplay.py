from dataclasses import dataclass, field
import random
from typing import Optional, Sequence

from mango.actions.abstract_actions import AbstractActions
from ..utils import Transition, ObsType, ActType


@dataclass(eq=False, slots=True, repr=True)
class ExperienceReplay:
    abs_actions: AbstractActions
    batch_size: int = 256
    capacity: int = 2**10
    last: int = field(default=0, init=False)
    memory: list[tuple[Transition, ...]] = field(default_factory=list, init=False)

    @property
    def size(self) -> int:
        return len(self.memory)

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        return self.size >= (batch_size or self.batch_size)

    def push(self, comand: ActType, transition: Transition) -> None:
        start_obs_masked = self.abs_actions.mask(transition.start_obs)
        next_obs_masked = self.abs_actions.mask(transition.next_obs)
        preprocessed_transitions = tuple(
            transition._replace(
                start_obs=start_obs_masked,
                next_obs=next_obs_masked,
                reward=self.abs_actions.compatibility(
                    comand, transition.start_obs, transition.next_obs
                ),
            )
            for comand in self.abs_actions.action_space
        )
        if self.size < self.capacity:
            self.memory.append(preprocessed_transitions)
        else:
            self.memory[self.last] = preprocessed_transitions
            self.last = (self.last + 1) % self.capacity

    def extend(self, comand: ActType, items: Sequence[Transition]) -> None:
        for item in items:
            self.push(comand, item)

    def sample(self, comand: ActType, quantity: Optional[int] = None) -> list[Transition]:
        if not self.can_sample(quantity):
            return []
        transition_tuples = random.sample(self.memory, (quantity or self.batch_size))
        return [transition[comand] for transition in transition_tuples]
