import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from contactnets.utils import utils

from contactnets.interaction import DirectInteraction
from contactnets.entity import Entity

import pdb

class DirectLearnable(DirectInteraction):
    interaction_module: Module

    def __init__(self, entities: List[Entity], interaction_module: Module) -> None:
        super(DirectLearnable, self).__init__(entities)
        self.interaction_module = interaction_module
    
    def compute_impulses(self, sp) -> List[Tensor]:
        module_in = None
        for entity in self.entities:
            if module_in is None:
                module_in = torch.cat((entity.get_state(), entity.get_control()), dim=1)
            else:
                module_in = torch.cat((module_in, entity.get_state(), entity.get_control()), dim=1)
        
        module_out = self.interaction_module(module_in.transpose(1,2)).transpose(1,2)
        
        impulses = []

        i = 0
        for entity in self.entities:
            impulses.append(module_out[:, i:i+entity.velocity_size(), :])
            i = i + entity.velocity_size()
        
        return impulses
