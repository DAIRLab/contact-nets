import pdb  # noqa
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module

from contactnets.entity import Entity
from contactnets.interaction import DirectLearnable
from contactnets.utils import quaternion as quat
from contactnets.utils import utils


class DeepStateAugment3D(Module):
    """Replace quat in [x,y,z,quat,vels,controls] with a vectorized rotation matrix."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        R = quat.quaternion_to_rotmat_vec(x[:, 3:7])
        x = torch.cat((x[:, 0:3], R, x[:, 7:]), dim=1)
        return x


class DeepLearnable(DirectLearnable):
    def __init__(self, entities: List[Entity], H = 128, depth = 2, data: Tensor = None) -> None:
        interaction = self.create_deep_net(H, depth, data)
        super().__init__(entities, interaction)

    def create_deep_net(self, H: int, depth: int, data: Optional[Tensor],
                        outsize: int = 6) -> Module:
        aug = DeepStateAugment3D()

        layers: List[Module] = []
        layers.append(aug)

        if data is None:
            normalizer = nn.Linear(24, 24)
            normalizer.weight = nn.Parameter(torch.eye(24), requires_grad=False)
            normalizer.bias = nn.Parameter(torch.zeros(24), requires_grad=False)
        else:
            data_batch_compressed = data.reshape(data.shape[0] * data.shape[1], -1)
            normalizer = utils.generate_normalizing_layer(aug(data_batch_compressed))
        layers.append(normalizer)

        layers.append(nn.Linear(24, H))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(H, H))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(H, outsize))

        return nn.Sequential(*layers)
