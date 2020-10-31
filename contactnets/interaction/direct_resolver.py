import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from contactnets.entity import Entity
from contactnets.interaction import InteractionResolver, Interaction

from contactnets.utils import tensor_utils, solver_utils

from typing import *

import pdb

class DirectResolver(InteractionResolver):
    def __init__(self, direct_interactions: ModuleList) -> None:
        super(DirectResolver, self).__init__([], direct_interactions)

    def step(self, system: 'System') -> None:
        self.step_impulses(system, self.compute_impulses(system))

    def compute_impulses(self, system: 'System') -> List[Tensor]:
        sp = system.params
        impulse_map = {}

        for direct in self.direct_interactions:
            impulses = direct.compute_impulses(sp)
            for entity, impulse in zip(direct.entities, impulses):
                if not entity in impulse_map:
                    impulse_map[entity] = impulse
                else:
                    impulse_map[entity] = impulse_map[entity] + impulse

        impulses = [None] * len(system.entities)
        for i, entity in enumerate(system.entities):
            impulses[i] = impulse_map[entity]

        return impulses
