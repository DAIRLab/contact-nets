import torch
from torch import Tensor, Size
from torch.nn import Module

from abc import ABC, abstractmethod

from contactnets.entity import Entity

from typing import *

class Ground2D(Entity):
    angle: Tensor
    height: Tensor

    def __init__(self, batch_n: int,
                       angle: Tensor = torch.tensor(0.0),
                       height: Tensor = torch.tensor(0.0)) -> None:
        super(Ground2D, self).__init__(0, 0)
        self.set_state(torch.zeros(batch_n, 0, 1), torch.zeros(batch_n, 0, 1))

        self.angle = angle
        self.height = height

    def compute_M(self, _) -> Tensor:
        return None

    def compute_M_i(self, _) -> Tensor:
        return None

    def compute_gamma(self, _) -> Tensor:
        return None

    def compute_f(self, _0, _1, velocity: Tensor, _2, dt=None) -> Tensor:
        return velocity
