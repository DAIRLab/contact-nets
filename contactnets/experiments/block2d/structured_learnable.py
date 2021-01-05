import math
import pdb  # noqa
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module

from contactnets.entity import Dynamic2D, Ground2D
from contactnets.interaction import PolyGround2D
from contactnets.jacprop import JacProp, Scale, StateAugment2D


class StructuredLearnable(PolyGround2D):
    phi_net: Module
    tangent_net: Module

    learn_normal: bool
    learn_tangent: bool

    def __init__(self, poly: Dynamic2D, ground: Ground2D, vertices: Tensor, mu: Tensor,
                 net_type: str, H: int, learn_normal: bool, learn_tangent: bool) -> None:
        super().__init__(poly, ground, vertices, mu)

        self.learn_normal = learn_normal
        self.learn_tangent = learn_tangent

        self.H = H

        if net_type == 'poly':
            self.init_polytope_phi_net()
            self.init_polytope_tangent_net()
        elif net_type == 'deep':
            self.init_deep_phi_net()
            self.init_deep_tangent_net()
        else:
            raise Exception('Network type not recognized')

        jacprop = JacProp()
        jacprop.jacobian_enable(self.phi_net)
        jacprop.jacobian_enable(self.tangent_net)

    def corner_tensor(self) -> Tensor:
        return torch.tensor([[0, 1, 1, -1],
                             [0, 1, 1, 1],
                             [0, 1, -1, 1],
                             [0, 1, -1, -1]]).double()

    def init_polytope_phi_net(self) -> None:
        state_aug = StateAugment2D(torch.tensor([0, math.pi / 2]))

        linear = nn.Linear(4, self.contact_n(), bias=False)
        linear.weight = nn.Parameter(self.corner_tensor(), requires_grad=True)
        with torch.no_grad():
            linear.weight.add_(torch.randn(linear.weight.size()) * 0.3)

        self.phi_net = nn.Sequential(state_aug, linear)

    def init_polytope_tangent_net(self) -> None:
        state_aug = StateAugment2D(torch.tensor([math.pi / 2, math.pi]))

        linear = nn.Linear(4, self.contact_n(), bias=False)
        linear.weight = nn.Parameter(self.corner_tensor(), requires_grad=True)

        with torch.no_grad():
            linear.weight.add_(torch.randn(linear.weight.size()) * 0.3)

        self.tangent_net = nn.Sequential(state_aug, linear)

    def init_deep_phi_net(self) -> None:
        self.phi_net = nn.Sequential(
            StateAugment2D(torch.tensor([0, math.pi / 2])),
            nn.Linear(4, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.contact_n(), bias=True),
            Scale(torch.tensor(5.0))
        )

    def init_deep_tangent_net(self) -> None:
        self.tangent_net = nn.Sequential(
            StateAugment2D(torch.tensor([math.pi / 2, math.pi])),
            nn.Linear(4, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.contact_n(), bias=True),
        )

    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_normal: return super().compute_phi(configurations)

        poly_configuration = configurations[0]

        return self.phi_net(poly_configuration.squeeze(-1)).unsqueeze(-1)

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_normal: return super().compute_Jn(configurations)

        poly_configuration = configurations[0]

        return JacProp.jacobian(self.phi_net, poly_configuration.squeeze(-1))

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_tangent: return super().compute_phi_t(configurations)

        poly_configuration = configurations[0]

        return self.tangent_net(poly_configuration.squeeze(-1)).unsqueeze(-1)

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        if not self.learn_tangent: return super().compute_Jt_tilde(configurations)

        poly_configuration = configurations[0]

        return JacProp.jacobian(self.tangent_net, poly_configuration.squeeze(-1))
