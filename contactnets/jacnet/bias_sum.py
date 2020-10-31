import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from contactnets.jacnet import JacModule

import pdb

class BiasSum(Module):
    jac_module: JacModule
    bias:       Parameter

    def __init__(self, jac_module: JacModule, bias: Parameter) -> None:
        super().__init__()

        self.jac_module = jac_module
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        batch_n = x.shape[0]
        return self.jac_module(x) + self.bias.unsqueeze(0).repeat(batch_n, 1, 1)

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        return self.jac_module.jac_func(operation_forwards)
