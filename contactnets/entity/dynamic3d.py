import torch
from torch import Tensor
from torch.nn import Module

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from contactnets.entity import Entity

from contactnets.utils import quaternion as quat

from typing import *

import pdb

@dataclass
class Dynamic3DParams:
    mass: Tensor = field(default_factory=lambda: torch.tensor(1.0))
    inertia: Tensor = field(default_factory=lambda: torch.tensor(1.0))

class Dynamic3D(Entity):
    params: Dynamic3DParams

    def __init__(self, configuration: Tensor, velocity: Tensor, params: Dynamic3DParams) -> None:
        super(Dynamic3D, self).__init__(7, 6)
        self.set_state(configuration, velocity)

        self.params = params

    def compute_M(self, _) -> Tensor:
        m = self.params.mass.unsqueeze(0)
        i = self.params.inertia.unsqueeze(0)
        return torch.diag(torch.cat((m, m, m, i, i, i))) \
            .unsqueeze(0).repeat(self.batch_n(), 1, 1)

    def compute_M_i(self, _) -> Tensor:
        m = self.params.mass.unsqueeze(0)
        i = self.params.inertia.unsqueeze(0)
        return torch.diag(torch.cat((1 / m, 1 / m, 1 / m, 1 / i, 1 / i, 1 / i))) \
            .unsqueeze(0).repeat(self.batch_n(), 1, 1)

    def compute_gamma(self, configuration: Tensor) -> Tensor:
        batch_n = self.batch_n()

        I = torch.eye(3).unsqueeze(0).repeat(batch_n, 1, 1)
        gamma_r = self.compute_gamma_r(configuration)

        r1 = torch.cat((I, torch.zeros(batch_n, 3, 3)), dim=2)
        r2 = torch.cat((torch.zeros(batch_n, 4, 3), gamma_r), dim=2)

        return torch.cat((r1, r2), dim=1)

    def compute_gamma_r(self, configuration: Tensor) -> Tensor:
        q = configuration[:, 3:7, 0]

        qmat = quat.qmat(q)
        # Remove column corresponding to zero-expanding w
        qmat = 0.5 * qmat[:, :, 1:4]
        return qmat

    def compute_f(self, sp, configuration: Tensor, velocity: Tensor,
                  control: Tensor, dt: Tensor = None) -> Tensor:
        if dt is None:
            dt = sp.dt.reshape(1, 1, 1).repeat(self.batch_n(), 1, 1)

        batch_n = self.batch_n()

        gravity_impulse = self.compute_M(configuration).bmm((-sp.g) * Tensor([0, 0, 1, 0, 0, 0]) \
                                            .reshape(1, 6, 1).repeat(batch_n, 1, 1)).bmm(dt)

        control_impulse = control.bmm(dt)

        return velocity + self.compute_M_i(configuration).bmm(control_impulse + gravity_impulse)
