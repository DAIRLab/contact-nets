import torch
from torch import Tensor
from torch.nn import Module

import pdb

class StateAugment2D(Module):
    # Replaces a state [x,y,theta] with [x,y,sin(phases + theta)]
    # Batching supported, and phases can have size >1
    phases: torch.Tensor

    def __init__(self, phases: Tensor) -> None:
        super(StateAugment2D, self).__init__()
        
        assert phases.dim() == 1
        self.phases = phases

    def forward(self, x: Tensor) -> Tensor:
        batch_n = x.shape[0]
        angles = x[:, :, 2:3]
        phases = self.phases.reshape(1, 1, -1).repeat(batch_n, 1, 1)
        
        # Outer sum of angles and phases
        shifted_angles = angles + phases
        sinusoids = torch.sin(shifted_angles)

        return torch.cat((x[:, :, 0:2], sinusoids), dim=2)

    def compute_jacobian(self, operation_forwards: Tensor) -> Tensor:
        batch_n = operation_forwards.shape[0]
        q_id = torch.eye(operation_forwards.shape[2])[0:2, :]
        q_id = q_id.unsqueeze(0).repeat(batch_n, 1, 1)

        angles = operation_forwards[:, :, 2:3]
        phases = self.phases.reshape(1, 1, -1).repeat(batch_n, 1, 1)
        shifted_angles = angles + phases
        sinusoids = torch.cos(shifted_angles)

        zeros = torch.zeros(batch_n, sinusoids.shape[2], 2)

        bottom = torch.cat((zeros, sinusoids.transpose(1,2)), dim=2)
        jac = torch.cat((q_id, bottom), dim=1)
        return jac
