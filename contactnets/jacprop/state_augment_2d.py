import pdb  # noqa

import torch
from torch import Tensor
from torch.nn import Module


class StateAugment2D(Module):
    """Replace a state [x,y,theta] with [x,y,sin(phases + theta)]."""
    phases: Tensor

    def __init__(self, phases: Tensor) -> None:
        super().__init__()

        assert phases.dim() == 1
        self.phases = phases

    def forward(self, x: Tensor) -> Tensor:
        batch_n = x.shape[0]
        angles = x[:, 2:3]
        phases = self.phases.reshape(1, -1).repeat(batch_n, 1)

        # Outer sum of angles and phases
        shifted_angles = angles + phases
        sinusoids = torch.sin(shifted_angles)

        return torch.cat((x[:, 0:2], sinusoids), dim=1)

    def compute_jacobian(self, forwards: Tensor) -> Tensor:
        batch_n = forwards.shape[0]
        q_id = torch.eye(forwards.shape[1])[0:2, :]
        q_id = q_id.unsqueeze(0).repeat(batch_n, 1, 1)

        angles = forwards[:, 2:3]
        phases = self.phases.reshape(1, -1).repeat(batch_n, 1)
        shifted_angles = angles + phases
        sinusoids = torch.cos(shifted_angles)

        zeros = torch.zeros(batch_n, sinusoids.shape[1], 2)

        bottom = torch.cat((zeros, sinusoids.unsqueeze(2)), dim=2)
        jac = torch.cat((q_id, bottom), dim=1)
        return jac
