from dataclasses import dataclass
import pdb  # noqa
from typing import List

import torch
from torch import Tensor

from contactnets.entity import Dynamic3D, Ground3D
from contactnets.interaction import Interaction
from contactnets.utils import utils


@dataclass
class PolyGeometry3D:
    """Represents the geometry of a 3D polygon.

    Attributes:
        vertices: vert_n x 3
    """
    vertices: Tensor


class PolyGround3D(Interaction):
    """A frictional interaction between a 3D polygon and a flat ground."""
    poly: Dynamic3D
    ground: Ground3D

    geometry: PolyGeometry3D
    mu: Tensor

    def __init__(self, poly: Dynamic3D, ground: Ground3D,
                 vertices: Tensor, mu: Tensor) -> None:
        super().__init__([poly, ground])

        self.poly = poly
        self.ground = ground

        self.geometry = PolyGeometry3D(vertices)
        self.mu = mu

    def ground_normal(self) -> Tensor: return torch.tensor([[0.0, 0, 1]])
    def ground_tangent(self) -> Tensor: return torch.tensor([[1.0, 0, 0], [0, 1, 0]])

    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return utils.transform_and_project_3d(
            poly_configuration, self.geometry.vertices, self.ground_normal())

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return utils.transform_and_project_3d_jacobian(
            poly_configuration, self.geometry.vertices, self.ground_normal())

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return self.mu * utils.transform_and_project_3d(
            poly_configuration, self.geometry.vertices, self.ground_tangent())

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        poly_configuration = configurations[0]

        return self.mu * utils.transform_and_project_3d_jacobian(
            poly_configuration, self.geometry.vertices, self.ground_tangent())

    def contact_n(self) -> int:
        return self.geometry.vertices.shape[0]
