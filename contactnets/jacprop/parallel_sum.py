import pdb  # noqa
from typing import cast

from torch import Tensor
from torch.nn import Module, ModuleList


class ParallelSum(Module):
    """Sum the outputs of multiple modules."""
    module_list: ModuleList

    def __init__(self, *modules: Module) -> None:
        super().__init__()

        assert len(modules) > 0

        self.module_list = ModuleList(list(modules))

    def forward(self, x: Tensor) -> Tensor:
        module_outs = [module(x) for module in self.module_list]
        return cast(Tensor, sum(module_outs))

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        module_jacobians = [module.compute_jacobian(forwards) for module in self.module_list]
        return cast(Tensor, sum(module_jacobians))
