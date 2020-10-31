import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from contactnets.utils import utils

from contactnets.interaction import PolyGround2D
from contactnets.entity import Dynamic2D
from contactnets.entity import Ground2D
from contactnets.jacnet import JacNet, JacModule, StateAugment2D, Series, Scale

import math

import pdb

class StructuredLearnable(PolyGround2D):
    phi_net: JacNet
    tangent_net: JacNet

    learn_normal: bool
    learn_tangent: bool

    def __init__(self, poly: Dynamic2D, ground: Ground2D,
                       vertices: Tensor, mu: Tensor, H: int,
                       learn_normal: bool, learn_tangent: bool) -> None:
        super(StructuredLearnable, self).__init__(poly, ground, vertices, mu)
        
        self.learn_normal = learn_normal
        self.learn_tangent = learn_tangent

        self.H = H
        if learn_normal and learn_tangent:
            #self.init_head_net()
            self.init_phi_net_deep()
            self.init_tangent_net_deep()
        elif learn_normal:
            self.init_phi_net_deep()
        elif learn_tangent:
            self.init_tangent_net_deep()

    def init_phi_net(self) -> None:
        state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        state_aug_module = JacModule.from_jac_enabled(state_aug)

        linear = torch.nn.Linear(4, self.k(), bias=False)
        linear.weight = torch.nn.Parameter(torch.tensor([[0, 1, 1, -1],
                                                         [0, 1, 1, 1],
                                                         [0, 1, -1, 1],
                                                         [0, 1, -1, -1]]).double(),
                                                         requires_grad=True)
        with torch.no_grad():
            linear.weight.add_(torch.randn(linear.weight.size()) * 0.3)

        linear_module = JacModule.from_linear(linear)

        modules = torch.nn.ModuleList([state_aug_module, linear_module]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_phi_net_deep(self) -> None:
        H = self.H
        state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        state_aug_module = JacModule.from_jac_enabled(state_aug)

        module1 = JacModule.from_linear(torch.nn.Linear(4, H))
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        module5 = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=True))

        scale = JacModule.from_jac_enabled(Scale(torch.tensor(5.0)))

        # modules = torch.nn.ModuleList([module1, module2, module3, module4, module5]) 
        modules = torch.nn.ModuleList([state_aug_module, module1, module2, module3, module4, module5, scale]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_phi_net_linear(self) -> None:
        state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        state_aug_module = JacModule.from_jac_enabled(state_aug)

        module1 = JacModule.from_linear(torch.nn.Linear(4, self.k(), bias=True))

        # modules = torch.nn.ModuleList([module1, module2, module3, module4, module5]) 
        modules = torch.nn.ModuleList([state_aug_module, module1]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_tangent_net_deep(self) -> None:
        H = self.H
        state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        state_aug_module = JacModule.from_jac_enabled(state_aug)

        module1 = JacModule.from_linear(torch.nn.Linear(4, H))
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        module5 = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=True))

        # modules = torch.nn.ModuleList([module1, module2, module3, module4, module5]) 
        modules = torch.nn.ModuleList([state_aug_module, module1, module2, module3, module4, module5]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_tangent_net_linear(self) -> None:
        state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        state_aug_module = JacModule.from_jac_enabled(state_aug)

        module1 = JacModule.from_linear(torch.nn.Linear(4, self.k(), bias=True))

        # modules = torch.nn.ModuleList([module1, module2, module3, module4, module5]) 
        modules = torch.nn.ModuleList([state_aug_module, module1]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_tangent_net(self) -> None:
        state_aug = StateAugment2D(torch.tensor([math.pi/2, math.pi]))
        state_aug_module = JacModule.from_jac_enabled(state_aug)

        linear = torch.nn.Linear(4, self.k(), bias=False)
        linear.weight = torch.nn.Parameter(torch.tensor([[1, 0, 1, -1],
                                                         [1, 0, 1, 1],
                                                         [1, 0, -1, 1],
                                                         [1, 0, -1, -1]]).double(),
                                                         requires_grad=True)

        with torch.no_grad():
            linear.weight.add_(torch.randn(linear.weight.size()) * 0.3)

        linear_module = JacModule.from_linear(linear)

        modules = torch.nn.ModuleList([state_aug_module, linear_module]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_head_net(self) -> None:
        H = self.H
        state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        state_aug_module = JacModule.from_jac_enabled(state_aug)

        module1 = JacModule.from_linear(torch.nn.Linear(4, H))
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        phi_head = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=True))
        tangent_head = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=True))
        
        scale = JacModule.from_jac_enabled(Scale(torch.tensor(5.0)))
        #phi_head.operation.weight = torch.nn.Parameter(phi_head.operation.weight * 10.0)
        # phi_head.operation.bias = torch.nn.Parameter(phi_head.operation.bias + 2.0)

        phi_modules = torch.nn.ModuleList([state_aug_module, module1, module2, module3, module4, phi_head, scale]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([state_aug_module, module1, module2, module3, module4, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def compute_phi_history(self, i=-1) -> Tensor:
        if not self.learn_normal:
            return super(StructuredLearnable, self).compute_phi_history(i=i)

        configuration = self.poly.get_configuration(i=i)
        
        return self.phi_net(configuration.transpose(1,2)).transpose(1,2) 

    def compute_Jn_history(self, i=-1) -> Tensor:
        if not self.learn_normal:
            return super(StructuredLearnable, self).compute_Jn_history(i=i)

        configuration = self.poly.get_configuration(i=i)
        
        return self.phi_net.jacobian(configuration.transpose(1,2)).transpose(1,2) 

    def compute_phi_t_history(self, i=-1) -> Tensor:
        if not self.learn_tangent:
            return super(StructuredLearnable, self).compute_phi_t_history(i=i)

        configuration = self.poly.get_configuration(i=i)
        
        return self.tangent_net(configuration.transpose(1,2)).transpose(1,2) 


    def compute_Jt_tilde_history(self, i=-1) -> Tensor:
        if not self.learn_tangent:
            return super(StructuredLearnable, self).compute_Jt_tilde_history(i=i)

        configuration = self.poly.get_configuration(i=i)
        
        return self.tangent_net.jacobian(configuration.transpose(1,2)).transpose(1,2) 
