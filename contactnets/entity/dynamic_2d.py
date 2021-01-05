from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from contactnets.entity import Entity

if TYPE_CHECKING:
    from contactnets.system import SystemParams


@dataclass
class Dynamic2DParams:
    """The parameters necessary to construct a 2D rigid body."""
    mass: Tensor = field(default_factory=lambda: torch.tensor(1.0))
    inertia: Tensor = field(default_factory=lambda: torch.tensor(1.0))


class Dynamic2D(Entity):
    """A 2D rigid body."""
    params: Dynamic2DParams

    def __init__(self, configuration: Tensor, velocity: Tensor,
                 params: Dynamic2DParams) -> None:
        super().__init__(3, 3)
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

    def compute_f(self, sp: 'SystemParams', configuration: Tensor, velocity: Tensor,
                  control: Tensor, dt: Tensor = None) -> Tensor:
        if dt is None:
            dt = sp.dt.reshape(1, 1, 1).repeat(self.batch_n(), 1, 1)

        gravity = (-sp.g) * Tensor([0, 1, 0]).reshape(1, 3, 1).repeat(self.batch_n(), 1, 1)
        gravity_impulse = self.compute_M(configuration).bmm(gravity).bmm(dt)

        control_impulse = control.bmm(dt)

        return velocity + self.compute_M_i(configuration).bmm(control_impulse + gravity_impulse)
