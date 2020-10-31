import torch
from torch import Tensor
from torch.nn import Module

from contactnets.utils import utils 
from contactnets.utils import quaternion as quat

import pdb

class VertexDistExtract3D(torch.nn.Module):
    extract_dirs: Tensor

    def __init__(self, extract_dirs: Tensor) -> Tensor:
        super().__init__()

        self.extract_dirs = extract_dirs

    def forward(self, x: Tensor) -> Tensor:
        batch_n = x.shape[0]

        ed = self.extract_dirs.unsqueeze(0)
        
        # x is batch_n x 1 x (7 + 3*k) tensor
        x = x.transpose(1,2)
        
        projections = []
        for i in range(batch_n):
            config = x[i:i+1, 0:7, :]
            verts  = x[i, 7:, :]
            assert verts.shape[0] % 3 == 0
            k = verts.shape[0] // 3
            
            #verts = verts.reshape(3, k)
            verts = verts.reshape(k, 3).t()

            projections.append(utils.transform_and_project_3d(config, verts, ed))

        return torch.cat(projections, dim=0)

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        x = operation_forwards.transpose(1,2)
        batch_n = x.shape[0]
        ed = self.extract_dirs.unsqueeze(0)
        
        jacobians = []
        for i in range(batch_n):
            config = x[i:i+1, 0:7, :]
            verts  = x[i, 7:, :]
            assert verts.shape[0] % 3 == 0
            k = verts.shape[0] // 3

            #verts = verts.reshape(3, k)
            verts = verts.reshape(k, 3).t()

            jacobians.append(utils.transform_and_project_3d_jacobian(
                config, verts, ed, vertex_jac=True))

        return torch.cat(jacobians, dim=0)

