import pdb  # noqa

from torch import Tensor
from torch.nn import Module

from contactnets.utils import utils


class TransformAndProject3D(Module):
    """Transform and project points along a set of vectors.

    Attributes:
        projections: proj_n x 3.
        points: point_n x 3.
    """
    projections: Tensor
    points: Tensor

    def __init__(self, projections: Tensor, points: Tensor) -> None:
        super().__init__()

        self.projections = projections
        self.points = points

    def forward(self, x: Tensor) -> Tensor:
        """Transform and project points in x along the projections member attribute.

        Args:
            x: batch_n x 7. The configuration.

        Returns:
            batch_n x (proj_n * points_n) tensor. Projections are interleaved for each point, as
            specified in transform_and_project_3d.
        """
        configuration = x.unsqueeze(-1)
        dists = utils.transform_and_project_3d(configuration, self.points, self.projections)
        return dists.squeeze(-1)

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        configuration = forwards.unsqueeze(-1)

        return utils.transform_and_project_3d_jacobian(
            configuration, self.points, self.projections)
