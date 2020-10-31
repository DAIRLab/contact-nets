import torch
from torch import Tensor
from torch.nn import Module
from contactnets.jacnet import JacModule
from contactnets.utils.quaternion import quaternion_to_rotmat_vec, quaternion_to_rotmat_jac
import pdb

class StateAugment3D(Module):
    # Replaces the quat in [x,y,z,quat] with a vectorized rotation matrix

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if x.nelement() > 7:
            if x.shape[0] == 1:
                x = x.transpose(0,1)
        R = quaternion_to_rotmat_vec(x[:,:,3:].squeeze(1)).unsqueeze(1)
        return torch.cat((x[:, :, 0:3], R), dim=2)

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        if operation_forwards.nelement() > 7:
            if operation_forwards.shape[0] == 1:
                operation_forwards = operation_forwards.transpose(0,1)
        batch_n = operation_forwards.shape[0]
        q_id = torch.eye(operation_forwards.shape[2])[0:3, :]
        q_id = q_id.unsqueeze(0).repeat(batch_n, 1, 1)

        squares_jac = quaternion_to_rotmat_jac(operation_forwards[:,:,3:].squeeze(1))

        zeros = torch.zeros(batch_n, 9, 3)

        bottom = torch.cat((zeros, squares_jac), dim=2)
        jac = torch.cat((q_id, bottom), dim=1)
        return jac
