import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from dataclasses import dataclass

from contactnets.utils import utils

from contactnets.interaction import Interaction
from contactnets.entity import Dynamic2D
from contactnets.entity import Ground2D

import math

import pdb

@dataclass
class PolyGeometry2D:
    # Shape is 2 x vertex_n
    vertices: Tensor
    angles: Tensor

class PolyGround2D(Interaction):
    poly: Dynamic2D
    ground: Ground2D

    geometry: PolyGeometry2D
    mu: Tensor

    def __init__(self, poly: Dynamic2D, ground: Ground2D,
                       vertices: Tensor, mu: Tensor) -> None:
        super(PolyGround2D, self).__init__([poly, ground])

        self.poly = poly
        self.ground = ground

        self.geometry = utils.create_geometry2d(vertices)
        self.mu = mu

    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        vertices_trans = utils.transform_vertices_2d(configuration, self.geometry.vertices)
        surf_trans = torch.cat((-torch.sin(self.ground.angle).reshape(1,1,1),
                                 torch.cos(self.ground.angle).reshape(1,1,1)), dim=2)
        surf_trans = surf_trans.repeat(self.batch_n(), 1, 1)
        phi = surf_trans.bmm(vertices_trans).transpose(1, 2)
        return phi - self.ground.height

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        Js = self.compute_corner_jacobians(configuration)
        Jns = []

        for J in Js:
            Jn_transform = torch.cat((-torch.sin(self.ground.angle).reshape(1,1,1),
                                      torch.cos(self.ground.angle).reshape(1,1,1)), dim=2)
            Jn_transform = Jn_transform.repeat(self.batch_n(), 1, 1)

            Jn = Jn_transform.bmm(J.transpose(1, 2))

            Jns.append(Jn.transpose(1, 2))

        Jn = torch.cat(tuple(Jns), dim=2).transpose(1, 2)
        return Jn

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        vertices_trans = utils.transform_vertices_2d(configuration, self.geometry.vertices)
        surf_trans = torch.cat((torch.cos(self.ground.angle).reshape(1,1,1),
                                torch.sin(self.ground.angle).reshape(1,1,1)), dim=2)
        surf_trans = surf_trans.repeat(self.batch_n(), 1, 1)
        phi_t = surf_trans.bmm(vertices_trans).transpose(1, 2)
        return phi_t

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        configuration = configurations[0] # Get configuration for poly
        Js = self.compute_corner_jacobians(configuration)
        Jts = []

        for J in Js:
            Jt_transform = torch.cat((torch.cos(self.ground.angle).reshape(1,1,1),
                                      torch.sin(self.ground.angle).reshape(1,1,1)), dim=2)
            Jt_transform = Jt_transform.repeat(self.batch_n(), 1, 1)

            Jt = Jt_transform.bmm(J.transpose(1, 2))

            Jts.append(Jt.transpose(1, 2))

        Jt = torch.cat(tuple(Jts), dim=2).transpose(1, 2)
        return Jt * self.mu

    def compute_corner_jacobians(self, configuration: Tensor) -> List[Tensor]:
        Js = []

        body_rot = configuration[:, 2:3, :].transpose(1, 2)

        for i, corner_angle in enumerate(self.geometry.angles):
            corner_angle_rep = corner_angle.repeat(self.batch_n()).reshape(-1, 1, 1)
            corner_rot = body_rot + corner_angle_rep
            angle_jacobian = torch.cat((-torch.sin(corner_rot), torch.cos(corner_rot)), dim=1)

            dist = torch.norm(self.geometry.vertices[:, i], 2)
            angle_jacobian = (angle_jacobian * dist).transpose(1,2)

            I = torch.eye(2).unsqueeze(0).repeat(self.batch_n(), 1, 1)
            J = torch.cat((I, angle_jacobian), dim=1)
            Js.append(J)

        return Js

    def k(self) -> int:
        return self.geometry.angles.numel()
