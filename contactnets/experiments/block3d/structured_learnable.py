import pdb  # noqa
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module

from contactnets.entity import Dynamic3D, Ground3D
from contactnets.interaction import PolyGround3D
from contactnets.jacprop import (BiasSum, JacProp, ModuleAugment, ParallelSum, Scale,
                                 StateAugment3D, TransformAndProject3D,
                                 TransformAndProjectInput3D)
from contactnets.utils import utils


class StructuredLearnable(PolyGround3D):
    phi_net: Module
    tangent_net: Module

    learn_normal: bool
    learn_tangent: bool

    def __init__(self, poly: Dynamic3D, ground: Ground3D, vertices: Tensor, mu: Tensor,
                 net_type: str, H: int, learn_normal: bool, learn_tangent: bool,
                 data: Tensor = None, polynoise: float = 0.4) -> None:
        super().__init__(poly, ground, vertices, mu)

        self.learn_normal = learn_normal
        self.learn_tangent = learn_tangent

        self.H = H
        self.data = data
        self.polynoise = polynoise

        if net_type == 'poly':
            self.init_polytope_net()
        elif net_type == 'deepvertex':
            self.init_deepvertex_net()
        elif net_type == 'deep':
            self.init_deep_net()
        else:
            raise Exception('Network type not recognized')

        jacprop = JacProp()
        jacprop.jacobian_enable(self.phi_net)
        jacprop.jacobian_enable(self.tangent_net)

    def corners(self) -> Tensor:
        return torch.tensor([[-1, -1, -1, -1, 1, 1, 1, 1],
                             [-1, -1, 1, 1, -1, -1, 1, 1],
                             [1, -1, -1, 1, 1, -1, -1, 1]]).t().double()

    def ground_normal(self) -> Tensor: return torch.tensor([[0.0, 0, 1]])
    def ground_tangent(self) -> Tensor: return torch.tensor([[1.0, 0, 0], [0, 1, 0]])

    def init_polytope_net(self) -> None:
        vertices_normal = nn.Parameter(self.corners(), requires_grad=True)
        vertices_tangent = nn.Parameter(self.corners(), requires_grad=True)

        zvec = nn.Parameter(self.ground_normal(), requires_grad=True)
        xyvecs = nn.Parameter(self.mu.item() * self.ground_tangent(), requires_grad=True)

        with torch.no_grad():
            vertices_normal.add_(torch.randn(vertices_normal.size()) * self.polynoise)
            vertices_tangent.add_(torch.randn(vertices_tangent.size()) * self.polynoise)

            zvec.add_(torch.randn(zvec.size()) * self.polynoise)
            xyvecs.add_(torch.randn(xyvecs.size()) * self.polynoise * self.mu.item())

        self.phi_net = TransformAndProject3D(zvec, vertices_normal)
        self.tangent_net = TransformAndProject3D(xyvecs, vertices_tangent)

    def init_deepvertex_net(self, learn_surface=False, pos_dependent=False) -> None:
        vertex_scale = 1.0
        vertices = nn.Parameter(vertex_scale * self.corners().reshape(24), requires_grad=False)
        with torch.no_grad():
            vertices.add_(torch.randn(vertices.size()) * self.polynoise)

        vertex_modules: List[Module] = []

        sa3d = StateAugment3D()
        if self.data is None:
            normalizer = nn.Linear(12, 12)
            normalizer.weight = nn.Parameter(torch.eye(12), requires_grad=False)
            normalizer.bias = nn.Parameter(torch.zeros(12), requires_grad=False)
        else:
            normalizer = utils.generate_normalizing_layer(sa3d(self.data).squeeze(1))
        vertex_modules.append(sa3d)
        vertex_modules.append(normalizer)

        if pos_dependent:
            vertex_modules.append(nn.Linear(12, self.H))
        else:
            # Zero out position contribution
            pos_mask_weight = torch.cat((torch.zeros(9, 3), torch.eye(9)), dim=1)
            pos_mask = nn.Linear(12, 9, bias=False)
            pos_mask.weight = nn.Parameter(pos_mask_weight, requires_grad=False)
            vertex_modules.append(pos_mask)
            vertex_modules.append(nn.Linear(9, self.H))

        vertex_modules.append(nn.Tanh())
        vertex_modules.append(nn.Linear(self.H, self.H))
        vertex_modules.append(nn.Tanh())
        vertex_modules.append(nn.Linear(self.H, 3 * self.contact_n()))
        vertex_modules.append(Scale(torch.tensor(1.0 / 3.0)))

        vertex_deep = nn.Sequential(*vertex_modules)
        vertex_net = BiasSum(vertex_deep, vertices)
        vertex_augment = ModuleAugment(vertex_net)

        # NORMAL NET
        projections = nn.Parameter(self.ground_normal(), requires_grad=learn_surface)
        with torch.no_grad():
            if learn_surface:
                projections.add_(torch.randn(projections.size()) * self.polynoise)

        vertex_project = TransformAndProjectInput3D(projections)
        self.phi_net = nn.Sequential(vertex_augment, vertex_project)

        # TANGENT NET
        mu = self.mu.item()
        projections = nn.Parameter(self.ground_tangent() * mu, requires_grad=learn_surface)
        with torch.no_grad():
            if learn_surface:
                projections.add_(torch.randn(projections.size()) * self.polynoise * mu)

        vertex_project = TransformAndProjectInput3D(projections)
        self.tangent_net = nn.Sequential(vertex_augment, vertex_project)

    def init_deep_net(self) -> None:
        # Has a deep component in parallel with the standard polytope portion

        vertices_normal = nn.Parameter(self.corners(), requires_grad=True)
        vertices_tangent = nn.Parameter(self.corners(), requires_grad=True)

        with torch.no_grad():
            vertices_normal.add_(torch.randn(vertices_normal.size()) * self.polynoise)
            vertices_tangent.add_(torch.randn(vertices_tangent.size()) * self.polynoise)

        projections = nn.Parameter(self.ground_normal(), requires_grad=True)
        with torch.no_grad():
            projections.add_(torch.randn(projections.size()) * self.polynoise)
        trans_and_project = TransformAndProject3D(projections, vertices_normal)
        self.phi_net = self.create_deep_net(True, trans_and_project)

        projections = nn.Parameter(self.mu.item() * self.ground_tangent(), requires_grad=True)
        with torch.no_grad():
            projections.add_(torch.randn(projections.size()) * self.polynoise * self.mu.item())
        trans_and_project = TransformAndProject3D(projections, vertices_tangent)
        self.tangent_net = self.create_deep_net(False, trans_and_project)

    def create_deep_net(self, normal, vert_net) -> Module:
        lin_out = self.contact_n() if normal else 2 * self.contact_n()

        deep_net = nn.Sequential(
            StateAugment3D(),
            nn.Linear(12, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.H),
            nn.Tanh(),
            nn.Linear(self.H, lin_out),
            Scale(torch.tensor(0.05))
        )

        return ParallelSum(deep_net, vert_net)


    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_normal: return super().compute_phi(configurations)

        poly_configuration = configurations[0].squeeze(-1)

        return self.phi_net(poly_configuration).unsqueeze(-1)

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_normal: return super().compute_Jn(configurations)

        poly_configuration = configurations[0].squeeze(-1)

        return JacProp.jacobian(self.phi_net, poly_configuration.squeeze(-1))

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_tangent: return super().compute_phi_t(configurations)

        poly_configuration = configurations[0].squeeze(-1)

        return self.tangent_net(poly_configuration).unsqueeze(-1)

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_tangent: return super().compute_Jt_tilde(configurations)

        poly_configuration = configurations[0].squeeze(-1)

        return JacProp.jacobian(self.tangent_net, poly_configuration.squeeze(-1))
