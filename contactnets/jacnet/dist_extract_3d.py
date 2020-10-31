import torch
from torch import Tensor
from torch.nn import Module

from contactnets.utils import utils 
from contactnets.utils import quaternion as quat

import pdb

class DistExtract3D(torch.nn.Module):
    extract_dirs: Tensor
    vertices: Tensor

    def __init__(self, extract_dirs: Tensor, vertices: Tensor) -> Tensor:
        super(DistExtract3D, self).__init__()

        self.extract_dirs = extract_dirs
        self.vertices = vertices 

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1,2)
        if x.nelement() > 7:
            if x.shape[0] == 1:
                x = x.transpose(0,2)

        ed = self.extract_dirs.unsqueeze(0)

        return utils.transform_and_project_3d(x, self.vertices, ed)

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        x = operation_forwards.transpose(1,2)
        if x.nelement() > 7:
            if x.shape[0] == 1:
                x = x.transpose(0,2)
        ed = self.extract_dirs.unsqueeze(0)

        return utils.transform_and_project_3d_jacobian(x, self.vertices, ed)
