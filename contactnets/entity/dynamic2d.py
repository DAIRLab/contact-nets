import torch
from torch import Tensor
from torch.nn import Module

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from contactnets.entity import Entity

from typing import *

import pdb

@dataclass
class Dynamic2DParams:
    mass: Tensor = field(default_factory=lambda: torch.tensor(1.0))
    inertia: Tensor = field(default_factory=lambda: torch.tensor(1.0))

class Dynamic2D(Entity):
    params: Dynamic2DParams

    def __init__(self, configuration: Tensor, velocity: Tensor, params: Dynamic2DParams) -> None:
        super(Dynamic2D, self).__init__(3, 3)
        self.set_state(configuration, velocity)

        self.params = params

    def compute_M(self, _) -> Tensor:
        m = self.params.mass.unsqueeze(0)
        i = self.params.inertia.unsqueeze(0)
        return torch.diag(torch.cat((m, m, i))) \
            .unsqueeze(0).repeat(self.batch_n(), 1, 1)

    def compute_M_i(self, _) -> Tensor:
        m = self.params.mass.unsqueeze(0)
        i = self.params.inertia.unsqueeze(0)
        return torch.diag(torch.cat((1 / m, 1 / m, 1 / i))) \
            .unsqueeze(0).repeat(self.batch_n(), 1, 1)

    def compute_gamma(self, _) -> Tensor:
        return torch.eye(3).unsqueeze(0).repeat(self.batch_n(), 1, 1)

    def compute_f(self, sp, configuration: Tensor, velocity: Tensor,
                  control: Tensor, dt: Tensor = None) -> Tensor:
        if dt is None:
            dt = sp.dt.reshape(1, 1, 1).repeat(self.batch_n(), 1, 1)

        gravity_impulse = self.compute_M(configuration).bmm((-sp.g) * Tensor([0, 1, 0]) \
                                .reshape(1, 3, 1).repeat(self.batch_n(), 1, 1)).bmm(dt)

        control_impulse = control.bmm(dt)

        return velocity + self.compute_M_i(configuration).bmm(control_impulse + gravity_impulse)
