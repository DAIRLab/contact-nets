import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from dataclasses import dataclass

from contactnets.utils import utils

from contactnets.interaction import Interaction
from contactnets.entity import Dynamic3D
from contactnets.entity import Ground3D

from contactnets.utils import quaternion as quat

import math

import pdb

@dataclass
class PolyGeometry3D:
    vertices: Tensor

class PolyGround3D(Interaction):
    poly: Dynamic3D
    ground: Ground3D

    geometry: PolyGeometry3D
    mu: Tensor

    def __init__(self, poly: Dynamic3D, ground: Ground3D,
            vertices: Tensor, mu: Tensor) -> None:
        super(PolyGround3D, self).__init__([poly, ground])

        self.poly = poly
        self.ground = ground

        self.geometry = PolyGeometry3D(vertices)
        self.mu = mu

    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        surf_trans = torch.tensor([0.0, 0, 1]).unsqueeze(0).unsqueeze(0)
        return utils.transform_and_project_3d(configuration, self.geometry.vertices, surf_trans).transpose(1,2)

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        surf_trans = torch.tensor([0.0, 0, 1]).unsqueeze(0).unsqueeze(0)
        Jn = utils.transform_and_project_3d_jacobian(configuration, self.geometry.vertices, surf_trans)
        return Jn

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        surf_trans_x = torch.tensor([1.0, 0, 0]).reshape(1, 1, 3)

        surf_trans_y = torch.tensor([0.0, 1, 0]).reshape(1, 1, 3)

        surf_trans = torch.cat((surf_trans_x,surf_trans_y),dim=1)
        return utils.transform_and_project_3d(configuration, self.geometry.vertices, surf_trans).transpose(1,2)

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        surf_trans_x = torch.tensor([1.0, 0, 0]).reshape(1, 1, 3)

        surf_trans_y = torch.tensor([0.0, 1, 0]).reshape(1, 1, 3)

        surf_trans = torch.cat((surf_trans_x,surf_trans_y),dim=1)
        Jt_tilde = self.mu * utils.transform_and_project_3d_jacobian(configuration, self.geometry.vertices, surf_trans)

        return Jt_tilde

    def compute_corner_jacobians(self, i=-1, configuration=None) -> List[Tensor]:
        if configuration is None:
            configuration = self.poly.configuration_history[i]
        batch_n = self.batch_n()
        k = self.k()

        Js = []

        if configuration.nelement() > 7:
            pdb.set_trace()
        body_rot = configuration[:, 3:7, 0]

        vertices_quat = torch.cat((torch.zeros(k, 1), self.geometry.vertices.t()), dim=1)

        for i, corner in enumerate(vertices_quat):
            corner = corner.unsqueeze(0)

            I = torch.eye(3).unsqueeze(0).repeat(batch_n, 1, 1)
            quat_jac = quat.qjac(body_rot, corner)
            J = torch.cat((I, quat_jac), dim=1)
            Js.append(J)

        return Js

    def k(self) -> int:
        return self.geometry.vertices.shape[1]
