from typing import Optional

import torch
from torch import Tensor

from contactnets.entity import Entity


class Ground2D(Entity):
    """A straight ground tilted at an angle with some offset."""
    angle: Tensor
    height: Tensor

    def __init__(self, batch_n: int,
                 angle: Tensor = torch.tensor(0.0),
                 height: Tensor = torch.tensor(0.0)) -> None:
        super().__init__(0, 0)
        self.set_state(torch.zeros(batch_n, 0, 1), torch.zeros(batch_n, 0, 1))

        self.angle = angle
        self.height = height

    def compute_M(self, _) -> Optional[Tensor]:
        return None

    def compute_M_i(self, _) -> Optional[Tensor]:
        return None

    def compute_gamma(self, _) -> Optional[Tensor]:
        return None

    def compute_f(self, _0, _1, velocity: Tensor, _2, dt=None) -> Tensor:
        return velocity
