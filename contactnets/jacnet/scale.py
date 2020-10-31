import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from typing import *

import pdb

class Scale(Module):
    scalar: Tensor 

    def __init__(self, scalar: Tensor) -> None:
        super(Scale, self).__init__()
        
        assert(scalar.dim() == 0)
        self.scalar = scalar

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scalar

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        n = operation_forwards.shape[2]

        return torch.eye(n).unsqueeze(0).repeat(operation_forwards.shape[0], 1, 1) * self.scalar
