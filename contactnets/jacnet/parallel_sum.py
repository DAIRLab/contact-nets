import torch
from torch import Tensor
from torch.nn import Module, ModuleList

import pdb

class ParallelSum(Module):
    jac_modules: ModuleList

    def __init__(self, jac_modules: ModuleList) -> None:
        super(ParallelSum, self).__init__()

        self.jac_modules = jac_modules 

    def forward(self, x: Tensor) -> Tensor:
        for i, module in enumerate(self.jac_modules):
            if i == 0:
                y = module(x)
            else:
                y = y + module(x)

        return y

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        for i, module in enumerate(self.jac_modules):
            if i == 0:
                jac = module.jac_func(operation_forwards)
            else:
                jac = jac + module.jac_func(operation_forwards)

        return jac
