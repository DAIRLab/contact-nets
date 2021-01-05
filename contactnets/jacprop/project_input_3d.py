import pdb  # noqa

import torch
from torch import Tensor
from torch.nn import Module

from contactnets.utils import utils


class TransformAndProjectInput3D(Module):
    """Transform and project points along vectors, where points are provided in the input.

    Attributes:
        projections: proj_n x 3.
    """
    projections: Tensor

    def __init__(self, projections: Tensor) -> None:
        super().__init__()

        self.projections = projections

    def reshape_points_vector(self, points: Tensor) -> Tensor:
        assert points.shape[0] % 3 == 0
        point_n = points.shape[0] // 3
        points = points.reshape(point_n, 3)

        return points

    def forward(self, x: Tensor) -> Tensor:
        """Transform and project points in x along the projections member attribute.

        Args:
            x: batch_n x (7 + 3 * points_n). points_n is the number of points, and the first
            7 entries is the configuration.

        Returns:
            batch_n x (proj_n * points_n) tensor. Projections are interleaved for each point, as
            specified in transform_and_project_3d.
        """
        def transform_and_project_batch(batch: Tensor) -> Tensor:
            config, points = batch[0:7].reshape(1, 7, 1), self.reshape_points_vector(batch[7:])

            return utils.transform_and_project_3d(config, points, self.projections)

        projections = [transform_and_project_batch(batch) for batch in x]

        return torch.cat(projections, dim=0).squeeze(2)

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        def transform_and_project_jacobian_batch(batch: Tensor) -> Tensor:
            config, points = batch[0:7].reshape(1, 7, 1), self.reshape_points_vector(batch[7:])

            return utils.transform_and_project_3d_jacobian(
                config, points, self.projections, vertex_jac=True)

        jacobians = [transform_and_project_jacobian_batch(batch) for batch in forwards]

        return torch.cat(jacobians, dim=0)
