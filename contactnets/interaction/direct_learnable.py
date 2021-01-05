import pdb  # noqa
from typing import TYPE_CHECKING, List

import torch
from torch import Tensor
from torch.nn import Module

from contactnets.entity import Entity
from contactnets.interaction import DirectInteraction

if TYPE_CHECKING:
    from contactnets.system import SystemParams


class DirectLearnable(DirectInteraction):
    """A wrapper for an interaction module that directly outputs impulses."""
    interaction: Module

    def __init__(self, entities: List[Entity], interaction: Module) -> None:
        super().__init__(entities)
        self.interaction = interaction

    def compute_impulses(self, sp: 'SystemParams') -> List[Tensor]:
        entity_ins = [torch.cat((e.get_state(), e.get_control()), dim=1) for e in self.entities]
        module_in = torch.cat(entity_ins, dim=1)
        module_out = self.interaction(module_in.squeeze(-1)).unsqueeze(-1)

        impulses = []

        i = 0
        for entity in self.entities:
            impulses.append(module_out[:, i:i + entity.velocity_size(), :])
            i += entity.velocity_size()

        return impulses
