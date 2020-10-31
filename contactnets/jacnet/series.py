import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from contactnets.jacnet import JacModule 

import pdb

class Series(Module):
    jac_modules: ModuleList

    def __init__(self, jac_modules: ModuleList) -> None:
        super(Series, self).__init__()

        self.jac_modules = jac_modules 

    def forward(self, x: Tensor) -> Tensor:
        for module in self.jac_modules:
            x = module(x)
        return x

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        x = operation_forwards
            
        for i, module in enumerate(self.jac_modules):
            if i == 0:
                jac = module.jac_func(x)
            else:
                jac = module.jac_func(x).bmm(jac)

            x = module(x)

        return jac
