import torch
from torch import Tensor
from torch.nn import Module

from contactnets.jacnet import JacModule

import pdb

class VertexAugment3D(Module):
    # Transforms a state [x,y,z,quat] with [x,y,z,quat,vertices]
    # vertices are produced by another jacmodule
    # Batching supported
    vertex_net: JacModule

    def __init__(self, vertex_net: JacModule) -> None:
        super().__init__()
        
        self.vertex_net = vertex_net

    def forward(self, x: Tensor) -> Tensor:
        batch_n = x.shape[0]
        vertices = self.vertex_net(x)

        return torch.cat((x, vertices), dim=2)

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        batch_n = operation_forwards.shape[0]
        q_id = torch.eye(7)
        q_id = q_id.unsqueeze(0).repeat(batch_n, 1, 1)

        vertices_jac = self.vertex_net.jac_func(self.vertex_net.operation_forwards)
        jac = torch.cat((q_id, vertices_jac), dim=1) 

        return jac
