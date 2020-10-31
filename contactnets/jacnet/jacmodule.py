import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from contactnets.utils import tensor_utils

import pdb

class JacModule(Module):
    # Wrapper for a pytorch module which remembers the forwards pass value
    jac_func: Callable[[Tensor], Tensor]

    operation: Module
    operation_forwards: Tensor

    def __init__(self, operation: Module,
                 jac_func: Callable[[Tensor], Tensor]) -> None:
        super(JacModule, self).__init__()

        self.jac_func = jac_func

        self.operation = operation
        self.operation_forwards = None

    @staticmethod
    def from_linear(operation: torch.nn.Linear) -> 'JacModule':
        jac_func = lambda forwards: operation.weight.unsqueeze(0).repeat(forwards.shape[0], 1, 1)
        return JacModule(operation, jac_func)
    
    @staticmethod
    def from_relu(operation: torch.nn.ReLU) -> 'JacModule':
        jac_func = lambda forwards: \
                tensor_utils.matrix_diag(torch.clamp(torch.sign(forwards.squeeze(1)), min=0))
        return JacModule(operation, jac_func)

    @staticmethod
    def from_tanh(operation: torch.nn.Tanh) -> 'JacModule':
        jac_func = lambda forwards: \
                tensor_utils.matrix_diag(1 - torch.tanh(forwards.squeeze(1)) ** 2) 
        return JacModule(operation, jac_func)

    @staticmethod
    def from_sin() -> 'JacModule':
        operation = lambda forwards: torch.sin(forwards)
        jac_func = lambda forwards: \
                tensor_utils.matrix_diag(torch.cos(forwards.squeeze(1)))
        return JacModule(operation, jac_func)

    @staticmethod
    def from_dropout(operation: torch.nn.Dropout) -> 'JacModule':
        # TODO: batch diagflats
        jac_func = lambda forwards: tensor_utils.matrix_diag(operation(torch.ones_like(forwards.squeeze(1))/(1.0 - operation.p*float(operation.training))))
        return JacModule(operation, jac_func)

    @staticmethod
    def from_jac_enabled(operation: Module) -> 'JacModule':
        # operation needs to have compute_jacobian function of correct signature
        # TODO: type check with mypy
        return JacModule(operation, operation.compute_jacobian)

    def forward(self, x: Tensor) -> Tensor:
        self.operation_forwards = x
        return self.operation(x)

    def jacobian(self, jac: Tensor) -> Tensor:
        # Jac comes backwards through the graph
        return self.jac_func(self.operation_forwards).transpose(1,2).bmm(jac)
