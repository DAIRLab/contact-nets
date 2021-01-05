import pdb  # noqa

from torch import Tensor
from torch.nn import Module, Parameter


class BiasSum(Module):
    """Add a bias to the output of a module."""
    module:     Module
    bias:       Parameter

    def __init__(self, module: Module, bias: Parameter) -> None:
        super().__init__()

        self.module = module
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        batch_n = x.shape[0]
        return self.module(x) + self.bias.unsqueeze(0).repeat(batch_n, 1)

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        return self.module.compute_jacobian(forwards)  # type: ignore
