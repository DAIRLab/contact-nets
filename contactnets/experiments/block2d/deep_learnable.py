from typing import List

import torch.nn as nn
from torch.nn import Module

from contactnets.entity import Entity
from contactnets.interaction import DirectLearnable


class DeepLearnable(DirectLearnable):
    def __init__(self, entities: List[Entity]) -> None:
        super().__init__(entities, self.create_deep_net())

    def create_deep_net(self) -> Module:
        sequential = nn.Sequential(
            nn.Linear(9, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 3)
        )

        return sequential
