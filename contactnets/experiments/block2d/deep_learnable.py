import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from contactnets.utils import utils

from contactnets.entity import Entity
from contactnets.interaction import DirectLearnable

import math

import pdb

class DeepLearnable(DirectLearnable):
    def __init__(self, entities: List[Entity]) -> None:
        super(DeepLearnable, self).__init__(entities, self.create_deep_net())

    def create_deep_net(self) -> Module:
        layers = []
        layers.append(torch.nn.Linear(9, 200))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(200, 200))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(200, 3))

        return torch.nn.Sequential(*layers)
