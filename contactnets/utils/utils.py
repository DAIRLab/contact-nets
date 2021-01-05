import math
import os
import pdb  # noqa
from typing import Any, Dict, List

import psutil
import torch
from torch import Tensor
import torch.nn as nn

from contactnets.utils import tensor_utils
import contactnets.utils.quaternion as quat


def elements_identical(li: List) -> bool:
    """Return true iff all elements of li are identical."""
    if len(li) == 0:
        return True

    return li.count(li[0]) == len(li)


def filter_none(li: List) -> List:
    """Remove all None elements from li."""
    return [i for i in li if (i is not None)]


def list_dict_swap(v: List[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    """Convert list of dicts to a dict of lists.

    >>> list_dict_swap([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]) == {'a': [1, 3], 'b': [2, 4]}
    """
    return {k: [dic[k] for dic in v] for k in v[0]}


def transpose_lists(li: List[List[Any]]) -> List[List[Any]]:
    """Transpose list of lists as if it were a matrix."""
    return list(map(list, zip(*li)))


def process_memory() -> float:
    """Return process memory usage in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss * 1e-6


def compute_quadratic_loss(A: Tensor, b: Tensor, c: Tensor, x: Tensor) -> Tensor:
    """Compute a quadratic with specified coefficients and variable."""
    return x.transpose(1, 2).bmm(A).bmm(x) + b.bmm(x) + c


def generate_normalizing_layer(data: Tensor) -> nn.Linear:
    """Create a linear layer which normalizes the input data Tensor.

    Args:
        data: batch_n x n

    Returns:
        A linear layer which normalizes each element [0, ..., n-1] along the batch_n dimension.
        Namely, layer(data).mean(dim=0) will be all zeros, and layer(data).std(dim=0) will be
        all ones. The only corner case is if all elements along a certain index are zero; i.e.
        data[i, j] = 0 for all i. Then layer(data) will have zero mean and standard deviation
        in that index. Note that layer(data).shape = data.shape.
    """
    means = data.mean(dim=0)
    stds = data.std(dim=0)

    stds_recip = 1 / stds
    stds_recip[stds_recip == float('Inf')] = 0

    layer = nn.Linear(data.shape[1], data.shape[1], bias=True)
    layer.weight = nn.Parameter(torch.diag(stds_recip), requires_grad=False)
    layer.bias = nn.Parameter(-means * stds_recip, requires_grad=False)
    return layer


def rot2d(theta: Tensor) -> Tensor:
    """Generate a batch of 2d rotation matrices from a batch of rotation angles.

    Args:
        theta: batch_n

    Returns:
        A tensor of the shape batch_n x 2 x 2.
    """
    assert theta.dim() == 1

    c, s = torch.cos(theta).reshape(-1, 1, 1), torch.sin(theta).reshape(-1, 1, 1)
    r1 = torch.cat((c, -s), dim=2)
    r2 = torch.cat((s, c), dim=2)
    rots = torch.cat((r1, r2), dim=1)

    return rots


################################################################################
#                              2D Transformations                              #
################################################################################


def transform_vertices_2d(configuration: Tensor, vertices: Tensor) -> Tensor:
    """Transform vertices by the state in configuration.

    Args:
        configuration: batch_n x 3 x 1. Second dimension represents x, y, theta.
        Last dimension just makes each batch entry a column vector.

        vertices: batch_n x vert_n x 2 OR vert_n x 2. If the latter the same vertices are used
        for every batch entry.

    Returns:
        A tensor of the shape batch_n x vert_n x 2.
    """
    batch_n = configuration.shape[0]

    if vertices.dim() == 2: vertices = vertices.unsqueeze(0).repeat(batch_n, 1, 1)
    assert vertices.shape[2] == 2
    vert_n = vertices.shape[1]

    rot = rot2d(configuration[:, 2, 0])
    trans = configuration[:, 0:2, :].repeat(1, 1, vert_n)

    vertices = torch.bmm(rot, vertices.transpose(1, 2)) + trans

    return vertices.transpose(1, 2)


def transform_and_project_2d(configuration: Tensor, vertices: Tensor,
                             projections: Tensor) -> Tensor:
    """Transform vertices by the configuration.

    Args:
        configuration: batch_n x 3 x 1. The second dimension represents x, y, theta.
        Last dimension just makes each batch entry a column vector.

        vertices: vert_n x 2.

        projections: proj_n x 2. Transformed vertices are projected along these vectors.

    Returns:
        A tensor of the shape batch_n x (vert_n * proj_n) x 1. Projections are interleaved
        along the second dimension. Meaning that we first stack proj_n projections for the
        first vertex, then proj_n projections for the second vertex, etc.
    """
    assert vertices.dim() == 2 and vertices.shape[1] == 2
    assert projections.dim() == 2 and projections.shape[1] == 2

    batch_n = configuration.shape[0]

    projections = projections.unsqueeze(0).repeat(batch_n, 1, 1)

    vertices = transform_vertices_2d(configuration, vertices)

    dists = projections.bmm(vertices.transpose(1, 2)).transpose(1, 2)

    return dists


def _compute_corner_jacobians(configuration: Tensor, vertices: Tensor) -> List[Tensor]:
    """Compute the jacobians of each corner position w.r.t. configuration."""
    batch_n = configuration.shape[0]
    corner_angles = -torch.atan2(vertices[:, 0], vertices[:, 1]) + math.pi / 2
    body_rot = configuration[:, 2:3, :].transpose(1, 2)

    Js = []

    for i, corner_angle in enumerate(corner_angles):
        corner_angle_rep = corner_angle.repeat(batch_n).reshape(-1, 1, 1)
        corner_rot = body_rot + corner_angle_rep
        angle_jacobian = torch.cat((-torch.sin(corner_rot), torch.cos(corner_rot)), dim=1)

        dist = torch.norm(vertices[i, :], 2)
        angle_jacobian = (angle_jacobian * dist).transpose(1, 2)

        Id = torch.eye(2).unsqueeze(0).repeat(batch_n, 1, 1)
        J = torch.cat((Id, angle_jacobian), dim=1)
        Js.append(J)

    return Js


def transform_and_project_2d_jacobian(configuration: Tensor, vertices: Tensor,
                                      projections: Tensor) -> Tensor:
    """Compute the Jacobian of the 2d transformation and projection.

    Args:
        configuration: batch_n x 3 x 1. The second dimension represents x, y, theta.
        Last dimension just makes each batch entry a column vector.

        vertices: vert_n x 2.

        projections: proj_n x 2. Transformed vertices are projected along these vectors.

    Returns:
        A tensor of the shape batch_n x (vert_n * proj_n) x 3. Projections are interleaved
        along the second dimension. Meaning that we first stack proj_n projection gradients
        for the first vertex, then proj_n project gradients for the second vertex, etc.
    """
    assert vertices.dim() == 2 and vertices.shape[1] == 2
    assert projections.dim() == 2 and projections.shape[1] == 2

    batch_n = configuration.shape[0]

    projections = projections.unsqueeze(0).repeat(batch_n, 1, 1)

    Js = _compute_corner_jacobians(configuration, vertices)
    projected_Js = [projections.bmm(J.transpose(1, 2)).transpose(1, 2) for J in Js]

    return torch.cat(tuple(projected_Js), dim=2).transpose(1, 2)


################################################################################
#                              3D Transformations                              #
################################################################################


def transform_vertices_3d(configuration: Tensor, vertices: Tensor) -> Tensor:
    """Transform vertices by the configuration.

    Args:
        configuration: batch_n x 7 x 1. The second dimension represents x, y, z, quaternion.
        Last dimension just makes each batch entry a column vector.

        vertices: batch_n x vert_n x 3 OR vert_n x 3. If the latter the same vertices are used
        for every batch entry.

    Returns:
        A tensor of the shape batch_n x vert_n x 3.
    """
    batch_n = configuration.shape[0]

    if vertices.dim() == 2: vertices = vertices.unsqueeze(0).repeat(batch_n, 1, 1)
    assert vertices.shape[2] == 3
    vertices = vertices.transpose(1, 2)
    vert_n = vertices.shape[2]

    vert_quat = torch.cat((torch.zeros(batch_n, vert_n, 1), vertices.transpose(1, 2)), dim=2)
    rot_quat = configuration[:, 3:7, :].squeeze(2)

    vert_quat = vert_quat.reshape(vert_n * batch_n, -1)
    rot_quat = rot_quat.repeat(1, vert_n).reshape(vert_n * batch_n, -1)

    vert_rot = quat.qmul(quat.qmul(rot_quat, vert_quat), quat.qinv(rot_quat))

    vert_rot = vert_rot.reshape(batch_n, vert_n, 4)

    vert_rot = vert_rot[:, :, 1:4]

    pos_shift = configuration[:, 0:3, :].transpose(1, 2)

    return vert_rot + pos_shift


def transform_and_project_3d(configuration: Tensor, vertices: Tensor,
                             projections: Tensor) -> Tensor:
    """Transform vertices by the configuration, then project along projections.

    Args:
        configuration: batch_n x 7 x 1. The second dimension represents x, y, z, quaternion.
        Last dimension just makes each batch entry a column vector.

        vertices: vert_n x 3.

        projections: proj_n x 3. Transformed vertices are projected along these vectors.

    Returns:
        A tensor of the shape batch_n x (vert_n * proj_n) x 1. Projections are interleaved
        along the second dimension. Meaning that we first stack proj_n projections for the
        first vertex, then proj_n projections for the second vertex, etc.
    """
    assert vertices.dim() == 2 and vertices.shape[1] == 3
    assert projections.dim() == 2 and projections.shape[1] == 3

    batch_n, vert_n, proj_n = configuration.shape[0], vertices.shape[0], projections.shape[0]

    projections = projections.unsqueeze(0).repeat(batch_n, 1, 1)

    vertices = transform_vertices_3d(configuration, vertices)

    dists = vertices.bmm(projections.transpose(1, 2))

    # Interleave the projections; should do nothing if proj_n = 1
    dists = dists.reshape(batch_n, proj_n * vert_n, 1)

    return dists


def transform_and_project_3d_jacobian(configuration: Tensor, vertices: Tensor,
                                      projections: Tensor, vertex_jac=False) -> Tensor:
    """Compute the Jacobian of the 3d transformation and projection w.r.t configuration.

    Args:
        configuration: batch_n x 7 x 1. The second dimension represents x, y, z, quaternion.
        Last dimension just makes each batch entry a column vector.

        vertices: vert_n x 3.

        projections: proj_n x 3. Transformed vertices are projected along these vectors.

        vertex_jac: indicates whether or not the Jacobian w.r.t. vertices should be added.

    Returns:
        For vertex_jac = False:
        A tensor of the shape batch_n x (vert_n * proj_n) x 7. Projections are interleaved
        along the second dimension. Meaning that we first stack proj_n projection gradients
        for the first vertex, then proj_n project gradients for the second vertex, etc.

        For vertex_jac = True:
        A tensor of the shape batch_n x (vert_n * proj_n) x (7 + 3 * vert_n). The original
        matrix is augmented with the jacobian of each projection w.r.t. vertex coordinates
        before transformation.
    """
    assert vertices.dim() == 2 and vertices.shape[1] == 3
    assert projections.dim() == 2 and projections.shape[1] == 3

    batch_n, vert_n, proj_n = configuration.shape[0], vertices.shape[0], projections.shape[0]

    projections = projections.unsqueeze(0).repeat(vert_n * batch_n, 1, 1)

    qrot = configuration[:, 3:7, 0]
    qverts = torch.cat((torch.zeros(vert_n, 1), vertices), dim=1)

    qrot = qrot.repeat(1, vert_n).reshape(vert_n * batch_n, -1)

    qverts = qverts.repeat(batch_n, 1)


    qjac = quat.qjac(qrot, qverts)
    qjac = qjac.reshape(-1, qjac.shape[1], qjac.shape[2])

    rot_jac_dist = projections.bmm(qjac.transpose(1, 2)).reshape(batch_n, vert_n * proj_n, -1)
    pos_jac_dist = projections.reshape(batch_n, vert_n * proj_n, -1)

    jac = torch.cat((pos_jac_dist, rot_jac_dist), dim=2)

    if vertex_jac:
        vertjac = quat.quaternion_to_rotmat(qrot)
        vertjac_dist = projections.bmm(vertjac).transpose(1, 2)

        vertjac_dist = tensor_utils.block_diag(vertjac_dist) \
            .t().unsqueeze(0).repeat(batch_n, 1, 1)
        jac = torch.cat((jac, vertjac_dist), dim=2)

    return jac
