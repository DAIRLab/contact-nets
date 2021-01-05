import pdb  # noqa
from typing import List

import torch
from torch import Tensor

from contactnets.entity import Dynamic2D, Ground2D
from contactnets.interaction import BallBin2D


class PairwiseStructuredLearnable(BallBin2D):
    def __init__(self, balls: List[Dynamic2D], walls: List[Ground2D],
                 radii: Tensor, ball_mus: Tensor, wall_mus: Tensor) -> None:
        super().__init__(balls, walls, radii, ball_mus, wall_mus)

        self.init_ballwall()

    def init_ballwall(self) -> None:
        self.Jn_vec = torch.nn.Parameter(torch.tensor([[[0., 1., 0.], [-1., 0., 0.]]]),
                                         requires_grad=True)
        with torch.no_grad():
            self.Jn_vec.add_(0.1 * torch.randn(self.Jn_vec.size()))

    def compute_Jn_ballwall(self, ground: Ground2D) -> Tensor:
        feature_vec = torch.cat((torch.cos(ground.angle).reshape(1, 1, 1),
                                 torch.sin(ground.angle).reshape(1, 1, 1)), dim=2)
        return feature_vec.bmm(self.Jn_vec).repeat(self.batch_n(), 1, 1)
