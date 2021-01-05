import pdb  # noqa

import torch
from torch import Tensor
from torch.nn import Module

import contactnets.utils.quaternion as quat


class StateAugment3D(Module):
    """Replace the quat in [x,y,z,quat] with a vectorized rotation matrix."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        R = quat.quaternion_to_rotmat_vec(x[:, 3:])
        return torch.cat((x[:, 0:3], R), dim=1)

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        forwards = forwards.unsqueeze(1)
        batch_n = forwards.shape[0]
        q_id = torch.eye(forwards.shape[2])[0:3, :]
        q_id = q_id.unsqueeze(0).repeat(batch_n, 1, 1)

        squares_jac = quat.quaternion_to_rotmat_jac(forwards[:, :, 3:].squeeze(1))

        zeros = torch.zeros(batch_n, 9, 3)

        bottom = torch.cat((zeros, squares_jac), dim=2)
        jac = torch.cat((q_id, bottom), dim=1)
        return jac
