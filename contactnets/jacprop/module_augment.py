import pdb  # noqa

import torch
from torch import Tensor
from torch.nn import Module


class ModuleAugment(Module):
    """Augments the input vector with the output of a module, evaluated with the input.

    So for the deep vertex module, the input is [x,y,z,quat], the vertex net is evaluated on the
    input, and the output is [x,y,z,quat,vertices].
    """
    module: Module

    def __init__(self, module: Module) -> None:
        super().__init__()

        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        vertices = self.module(x)

        return torch.cat((x, vertices), dim=1)

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        batch_n = forwards.shape[0]
        q_id = torch.eye(forwards.shape[1]).unsqueeze(0).repeat(batch_n, 1, 1)
        vertices_jac = self.module.compute_jacobian(forwards)  # type: ignore
        jac = torch.cat((q_id, vertices_jac), dim=1)

        return jac
