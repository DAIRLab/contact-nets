import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from dataclasses import dataclass

from contactnets.utils import utils

from contactnets.interaction import PolyGround3D
from contactnets.entity import Dynamic3D
from contactnets.entity import Ground3D
from contactnets.jacnet import JacNet, JacModule, DistExtract3D, Series, StateAugment3D, Scale, ParallelSum, VertexAugment3D, VertexDistExtract3D, BiasSum

import math

import pdb

class StructuredLearnable(PolyGround3D):
    phi_net: JacNet

    learn_normal: bool
    learn_tangent: bool

    def __init__(self, poly: Dynamic3D, ground: Ground3D, polytope,
            vertices: Tensor, mu: Tensor, H: int, learn_normal: bool, learn_tangent: bool, data: Tensor = None) -> None:
        super(StructuredLearnable, self).__init__(poly, ground, vertices, mu)

        self.learn_normal = learn_normal
        self.learn_tangent = learn_tangent
        self.H = H
        self.data = data

        if polytope:
            self.init_phi_net_polytope()
            self.init_tangent_net_polytope()
        else:
            self.init_phi_net_deep()
            self.init_tangent_net_deep()

    def init_phi_net_polytope(self) -> None:
        vertices = torch.nn.Parameter(torch.tensor([[1, -1, 1],
                                                    [1, 1, 1],
                                                    [-1, 1, 1],
                                                    [-1, -1, 1],
                                                    [1, -1, -1],
                                                    [1, 1, -1],
                                                    [-1, 1, -1],
                                                    [-1, -1, -1]]).double().t(),
                                                    requires_grad=True)

        zvec = torch.nn.Parameter(torch.tensor([[0.0, 0, 1]]), requires_grad=True)

        with torch.no_grad():
            vertices.add_(torch.randn(vertices.size()) * 0.4)
            zvec.add_(torch.randn(zvec.size()) * 0.4)
        
        module1 = DistExtract3D(zvec, vertices)

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(torch.nn.ModuleList([JacModule.from_jac_enabled(module1)]))))

    def init_tangent_net_polytope(self) -> None:
        vertices = torch.nn.Parameter(torch.tensor([[1, -1, 1],
                                                    [1, 1, 1],
                                                    [-1, 1, 1],
                                                    [-1, -1, 1],
                                                    [1, -1, -1],
                                                    [1, 1, -1],
                                                    [-1, 1, -1],
                                                    [-1, -1, -1]]).double().t(),
                                                    requires_grad=True)
        projvecs = torch.nn.Parameter(self.mu.item() * torch.tensor([[1.0, 0, 0],
                                                                     [0.0, 1, 0]]), requires_grad=True)

        with torch.no_grad():
            vertices.add_(torch.randn(vertices.size()) * 0.4)
            projvecs.add_(torch.randn(projvecs.size()) * 0.4 * self.mu.item())
        
        module1 = DistExtract3D(projvecs, vertices)

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(torch.nn.ModuleList([JacModule.from_jac_enabled(module1)]))))

    def create_deep_net(self, normal) -> None:
        H = self.H

        vertices = torch.nn.Parameter(torch.tensor([[1, -1, 1],
                                                    [1, 1, 1],
                                                    [-1, 1, 1],
                                                    [-1, -1, 1],
                                                    [1, -1, -1],
                                                    [1, 1, -1],
                                                    [-1, 1, -1],
                                                    [-1, -1, -1]]).double().t(),
                                                    requires_grad=True)

        with torch.no_grad():
            vertices.add_(torch.randn(vertices.size()) * 0.4)

        if normal:
            projvecs = torch.nn.Parameter(torch.tensor([[0.0, 0, 1]]), requires_grad=True)
            with torch.no_grad():
                projvecs.add_(torch.randn(projvecs.size()) * 0.4)
        else:
            projvecs = torch.nn.Parameter(self.mu.item() * \
                    torch.tensor([[1.0, 0, 0], [0.0, 1, 0]]), requires_grad=True)
            with torch.no_grad():
                projvecs.add_(torch.randn(projvecs.size()) * 0.4 * self.mu.item())
        
        module1 = DistExtract3D(projvecs, vertices)

        vertnet = JacModule.from_jac_enabled(Series(torch.nn.ModuleList([JacModule.from_jac_enabled(module1)])))

        deep_modules = torch.nn.ModuleList([])
        deep_modules.append(JacModule.from_jac_enabled(StateAugment3D()))
        deep_modules.append(JacModule.from_linear(torch.nn.Linear(12, H)))
        deep_modules.append(JacModule.from_tanh(torch.nn.Tanh()))
        deep_modules.append(JacModule.from_linear(torch.nn.Linear(H, H)))
        deep_modules.append(JacModule.from_tanh(torch.nn.Tanh()))

        if normal:
            deep_modules.append(JacModule.from_linear(torch.nn.Linear(H, self.k())))
        else:
            deep_modules.append(JacModule.from_linear(torch.nn.Linear(H, 2 * self.k())))

        deep_modules.append(JacModule.from_jac_enabled(Scale(torch.tensor(0.05))))
        deepnet = JacModule.from_jac_enabled(Series(deep_modules))

        parallel_modules = torch.nn.ModuleList([vertnet, deepnet])

        return JacNet(JacModule.from_jac_enabled(ParallelSum(parallel_modules)))

    def init_phi_net_deep(self) -> None:
        self.phi_net = self.create_deep_net(normal=True)

    def init_tangent_net_deep(self) -> None:
        self.tangent_net = self.create_deep_net(normal=False)

    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_normal:
            return super().compute_phi(configurations)

        configuration = configurations[0] # Get configuration for poly

        return self.phi_net(configuration.transpose(1,2)).transpose(1,2)

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_normal:
            return super().compute_Jn(configurations)

        configuration = configurations[0] # Get configuration for poly

        return self.phi_net.jacobian(configuration.transpose(1,2)).transpose(1,2)

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_tangent:
            return super().compute_phi_t(configurations)

        configuration = configurations[0] # Get configuration for poly

        return self.tangent_net(configuration.transpose(1,2)).transpose(1,2)

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_tangent:
            return super().compute_Jt_tilde(configurations)

        configuration = configurations[0] # Get configuration for poly

        return self.tangent_net.jacobian(configuration.transpose(1,2)).transpose(1,2)
