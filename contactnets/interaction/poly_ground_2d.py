from dataclasses import dataclass
import pdb  # noqa
from typing import List

import torch
from torch import Tensor

from contactnets.entity import Dynamic2D, Ground2D
from contactnets.interaction import Interaction
from contactnets.utils import utils


@dataclass
class PolyGeometry2D:
    """Represents the geometry of a 2D polygon.

    Attributes:
        vertices: vert_n x 2
    """
    vertices: Tensor


class PolyGround2D(Interaction):
    """A frictional interaction between a 2D polygon and a flat ground."""
    poly: Dynamic2D
    ground: Ground2D

    geometry: PolyGeometry2D
    mu: Tensor

    def __init__(self, poly: Dynamic2D, ground: Ground2D, vertices: Tensor, mu: Tensor) -> None:
        super().__init__([poly, ground])

        self.poly = poly
        self.ground = ground

        self.geometry = PolyGeometry2D(vertices)
        self.mu = mu

    def ground_angle_sin(self) -> Tensor: return torch.sin(self.ground.angle).reshape(1, 1)
    def ground_angle_cos(self) -> Tensor: return torch.cos(self.ground.angle).reshape(1, 1)

    def ground_normal(self) -> Tensor:
        return torch.cat((-self.ground_angle_sin(), self.ground_angle_cos()), dim=1)

    def ground_tangent(self) -> Tensor:
        return torch.cat((self.ground_angle_cos(), self.ground_angle_sin()), dim=1)


    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return utils.transform_and_project_2d(
            poly_configuration, self.geometry.vertices, self.ground_normal())

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return utils.transform_and_project_2d_jacobian(
            poly_configuration, self.geometry.vertices, self.ground_normal())

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return self.mu * utils.transform_and_project_2d(
            poly_configuration, self.geometry.vertices, self.ground_tangent())

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return self.mu * utils.transform_and_project_2d_jacobian(
            poly_configuration, self.geometry.vertices, self.ground_tangent())

    def contact_n(self) -> int:
        return self.geometry.vertices.shape[0]
