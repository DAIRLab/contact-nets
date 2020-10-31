import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from contactnets.utils import utils
from contactnets.utils import quaternion as quat

from contactnets.entity import Entity
from contactnets.interaction import DirectLearnable

import math

import pdb

class DeepStateAugment3D(Module):
    # Replaces the quat in [x,y,z,quat, vels, controls] with a vectorized rotation matrix

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        R = quat.quaternion_to_rotmat_vec(x[:,:,3:7].squeeze(0)).unsqueeze(0)
        x = torch.cat((x[:, :, 0:3], R, x[:, :, 7:]), dim=2)
        return x


class DeepLearnable(DirectLearnable):
    def __init__(self, entities: List[Entity], H = 128, depth = 2,
                 data: Tensor = None) -> None:
        super().__init__(entities, self.create_deep_net(H, depth, data))

    def create_deep_net(self, H: int, depth: int, data: Tensor) -> Module:
        aug = DeepStateAugment3D()

        normalizer = utils.generate_normalizing_layer(aug(data).squeeze(0))

        layers = []
        layers.append(aug)
        layers.append(normalizer)
        layers.append(torch.nn.Linear(24, H))
        layers.append(torch.nn.ReLU())
        for _ in range(depth-1):
            layers.append(torch.nn.Linear(H, H))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(H, 6))

        return torch.nn.Sequential(*layers)
