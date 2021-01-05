from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, List

from torch import Tensor
from torch.nn import Module

from contactnets.entity import Entity
from contactnets.interaction import InteractionResolver

if TYPE_CHECKING:
    from contactnets.system import System


class DirectResolver(InteractionResolver):
    """A resolver that sums the predicted impulses of several direct interactions."""
    def __init__(self, direct_interactions: List[Module]) -> None:
        super().__init__([], direct_interactions)

    def step(self, system: 'System') -> None:
        self.step_impulses(system, self.compute_impulses(system))

    def compute_impulses(self, system: 'System') -> List[Tensor]:
        # A dictionary of accumulated impulses for each entity
        impulse_dict: DefaultDict[Entity, List[Tensor]] = defaultdict(list)

        for interaction in self.direct_interactions:
            impulses = interaction.compute_impulses(system.params)
            for entity, impulse in zip(interaction.entities, impulses):
                impulse_dict[entity].append(impulse)

        impulses = [sum(impulse_dict[entity]) for entity in system.entities]
        return impulses
