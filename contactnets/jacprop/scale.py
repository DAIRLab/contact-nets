import pdb  # noqa

import torch
from torch import Tensor
from torch.nn import Module


class Scale(Module):
    """Scale input by a scalar tensor."""
    scalar: Tensor

    def __init__(self, scalar: Tensor) -> None:
        super().__init__()

        assert(scalar.dim() == 0)
        self.scalar = scalar

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scalar

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        batch_n = forwards.shape[0]
        n = forwards.shape[1]

        return torch.eye(n).unsqueeze(0).repeat(batch_n, 1, 1) * self.scalar
