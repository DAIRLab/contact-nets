import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from contactnets.jacnet import JacModule 

import pdb

class JacNet(Module):
    root: JacModule

    def __init__(self, root: JacModule) -> None:
        super(JacNet, self).__init__()

        self.root = root

    def forward(self, x: Tensor) -> Tensor:
        return self.root(x)

    def jacobian(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            val = self.forward(x)
        
        batch_n = val.shape[0]
        out_size = val.shape[2]
        jac = torch.eye(out_size).unsqueeze(0).repeat(batch_n, 1, 1)
        
        return self.root.jacobian(jac)
