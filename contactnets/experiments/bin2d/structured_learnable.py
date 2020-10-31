import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from dataclasses import dataclass

from contactnets.utils import utils

from contactnets.interaction import PolyGround3D
from contactnets.entity import Dynamic3D
from contactnets.entity import Ground3D
from contactnets.jacnet import JacNet, JacModule, DistExtract3D, QuadSum3D, Series, StateAugment3D, VertexAugment3D, VertexDistExtract3D, Scale

import math

import pdb

class StructuredLearnable(PolyGround3D):
    phi_net: JacNet

    learn_normal: bool
    learn_tangent: bool

    def __init__(self, poly: Dynamic3D, ground: Ground3D,
            vertices: Tensor, mu: Tensor, H: int, learn_normal: bool, learn_tangent: bool, data: Tensor = None) -> None:
        super(StructuredLearnable, self).__init__(poly, ground, vertices, mu)
        
        self.learn_normal = learn_normal
        self.learn_tangent = learn_tangent
        self.H = H
        self.data = data
        #self.init_phi_net_deep_quad()
        #self.init_head_net()

        self.init_split_net()
        #self.init_split_net_quat()
        #self.init_split_net_quat()

        #self.init_head_net_siren()
        #self.init_head_net_siren_quat()
        #self.init_head_net_shrunk()
        #self.init_linear()
        #self.init_head_net_thin_siren()
        #self.init_head_net_siren_deeper()
        #self.init_linear()
        #self.init_vertex()
        #self.init_tangent_net()

        #if learn_normal and learn_tangent:
        #    self.init_head_net()
        #elif learn_normal:
        #    self.init_phi_net_deep()
        #elif learn_tangent:
        #   self.init_tangent_net_deep()

        # self.init_phi_net()
        # self.init_tangent_net()
        # self.init_phi_net_deep()

        # self.init_phi_net_rotinv()
        # self.init_tangent_net_deep()

    def init_head_net(self) -> None:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        module1 = JacModule.from_linear(torch.nn.Linear(3 + 9, H))
        #module2 = JacModule.from_relu(torch.nn.ReLU())
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        #module2 = JacModule.from_sin()
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        #module4 = JacModule.from_relu(torch.nn.ReLU())
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        #module4 = JacModule.from_sin()
        phi_head = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=False))
        THL = torch.nn.Linear(H, 2 * self.k(), bias=False)
        #THL.weight = torch.nn.Parameter(torch.zeros(H, 2 * self.k()).t())
        #THL.weight.requires_grad = False
        tangent_head = JacModule.from_linear(THL)

        phi_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, phi_head]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def single_net(self, out_dim, has_bias, scaling, rotmat: bool = True, linear=True) -> JacNet:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        if rotmat:
            sa3d = StateAugment3D()
            normalizer = self.generate_normalizing_layer(sa3d(self.data).squeeze(1))
            module0 = JacModule.from_jac_enabled(sa3d)
        else:
            normalizer = self.generate_normalizing_layer(self.data.squeeze(0))
            #pdb.set_trace()
        extra = 9 if rotmat else 4

        scale = JacModule.from_jac_enabled(Scale(torch.tensor(scaling)))
        if linear:
            module1 = JacModule.from_linear(torch.nn.Linear(3 + extra, out_dim))
            if rotmat:
                phi_modules = torch.nn.ModuleList([module0, module1, scale])
            else:
                phi_modules = torch.nn.ModuleList([module1, scale])
        else:
            module1 = JacModule.from_linear(torch.nn.Linear(3 + extra, H))
            module2 = JacModule.from_tanh(torch.nn.Tanh())
            module3 = JacModule.from_linear(torch.nn.Linear(H, H))
            module4 = JacModule.from_tanh(torch.nn.Tanh())
            phi_head = JacModule.from_linear(torch.nn.Linear(H, out_dim, bias=has_bias))

            if rotmat:
                phi_modules = torch.nn.ModuleList([module0, normalizer, module1, module2, module3, module4, phi_head, scale])
            else:
                phi_modules = torch.nn.ModuleList([normalizer, module1, module2, module3, module4, phi_head, scale])

        return JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

    def init_split_net(self) -> None:
        self.phi_net = self.single_net(self.k(), True, 5.0, rotmat=True, linear=False)

        self.tangent_net = self.single_net(2 * self.k(), False, 1.0, rotmat=True, linear=False)

    def init_split_net_quat(self) -> None:

        self.phi_net = self.single_net(self.k(), True, 5.0, False)

        self.tangent_net = self.single_net(2 * self.k(), False, 1.0, False)

    def generate_normalizing_layer(self, data):
        #pdb.set_trace()
        means = data.mean(dim=0)
        stds = data.std(dim=0)
        layer = torch.nn.Linear(data.shape[1], data.shape[1], bias=True)
        layer.weight = torch.nn.Parameter(torch.diag(1/stds), requires_grad=False)
        layer.bias = torch.nn.Parameter(-means / stds, requires_grad=False)
        return JacModule.from_linear(layer)

    def init_head_net_shrunk(self) -> None:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        l1 = torch.nn.Linear(3 + 9, H)
        torch.nn.init.kaiming_uniform_(l1.weight, mode='fan_in')
        l1.weight = torch.nn.Parameter(math.sqrt(0.01) * l1.weight, requires_grad=True)
        l1.bias = torch.nn.Parameter(math.sqrt(0.01) * l1.bias, requires_grad=True)
        module1 = JacModule.from_linear(l1)

        module2 = JacModule.from_tanh(torch.nn.Tanh())

        l2 = torch.nn.Linear(H, H)
        torch.nn.init.kaiming_uniform_(l2.weight, mode='fan_in')
        l2.weight = torch.nn.Parameter(math.sqrt(0.1) * l2.weight, requires_grad=True)
        l2.bias = torch.nn.Parameter(math.sqrt(0.1) * l2.bias, requires_grad=True)
        module3 = JacModule.from_linear(l2)

        module4 = JacModule.from_tanh(torch.nn.Tanh())

        lphi = torch.nn.Linear(H, self.k(), bias=False)
        phi_head = JacModule.from_linear(lphi)

        ltan = torch.nn.Linear(H, 2 * self.k(), bias=False)
        tangent_head = JacModule.from_linear(ltan)

        phi_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, phi_head]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_head_net_siren_quat(self) -> None:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        DATA_SCALE = 3.
        l1 = torch.nn.Linear(3 + 4, H)
        torch.nn.init.kaiming_uniform_(l1.weight, mode='fan_in', nonlinearity='relu')
        l1.weight = torch.nn.Parameter(DATA_SCALE * l1.weight, requires_grad=True)
        module1 = JacModule.from_linear(l1)
        #module2 = JacModule.from_relu(torch.nn.ReLU())
        #module2 = JacModule.from_tanh(torch.nn.Tanh())
        module2 = JacModule.from_sin()
        l2 = torch.nn.Linear(H, H)
        torch.nn.init.kaiming_uniform_(l2.weight, mode='fan_in', nonlinearity='relu')
        #l2.weight = torch.nn.Parameter(math.sqrt(3) * l2.weight, requires_grad=True)
        module3 = JacModule.from_linear(l2)
        #module4 = JacModule.from_relu(torch.nn.ReLU())
        #module4 = JacModule.from_tanh(torch.nn.Tanh())
        module4 = JacModule.from_sin()

        lphi = torch.nn.Linear(H, self.k())#, bias=False)
        torch.nn.init.kaiming_uniform_(lphi.weight, mode='fan_in', nonlinearity='relu')
        #lphi.weight = torch.nn.Parameter(math.sqrt(3) * lphi.weight,requires_grad=True)
        phi_head = JacModule.from_linear(lphi)

        ltan = torch.nn.Linear(H, 2 * self.k(), bias=False)
        torch.nn.init.kaiming_uniform_(ltan.weight, mode='fan_in', nonlinearity='relu')
        #ltan.weight = torch.nn.Parameter(math.sqrt(3) * ltan.weight,requires_grad=True)
        tangent_head = JacModule.from_linear(ltan)

        phi_modules = torch.nn.ModuleList([module1, module2, module3, module4, phi_head]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([module1, module2, module3, module4, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_head_net_siren(self) -> None:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        DATA_SCALE = 1.
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        l1 = torch.nn.Linear(3 + 9, H)
        torch.nn.init.kaiming_uniform_(l1.weight, mode='fan_in', nonlinearity='relu')
        l1.weight = torch.nn.Parameter(DATA_SCALE * l1.weight, requires_grad=True)
        module1 = JacModule.from_linear(l1)
        #module2 = JacModule.from_relu(torch.nn.ReLU())
        #module2 = JacModule.from_tanh(torch.nn.Tanh())
        module2 = JacModule.from_sin()
        l2 = torch.nn.Linear(H, H)
        torch.nn.init.kaiming_uniform_(l2.weight, mode='fan_in', nonlinearity='relu')
        #l2.weight = torch.nn.Parameter(math.sqrt(3) * l2.weight, requires_grad=True)
        module3 = JacModule.from_linear(l2)
        #module4 = JacModule.from_relu(torch.nn.ReLU())
        #module4 = JacModule.from_tanh(torch.nn.Tanh())
        module4 = JacModule.from_sin()

        lphi = torch.nn.Linear(H, self.k())#, bias=False)
        torch.nn.init.kaiming_uniform_(lphi.weight, mode='fan_in', nonlinearity='relu')
        #lphi.weight = torch.nn.Parameter(math.sqrt(3) * lphi.weight,requires_grad=True)
        phi_head = JacModule.from_linear(lphi)

        ltan = torch.nn.Linear(H, 2 * self.k(), bias=False)
        torch.nn.init.kaiming_uniform_(ltan.weight, mode='fan_in', nonlinearity='relu')
        #ltan.weight = torch.nn.Parameter(math.sqrt(3) * ltan.weight,requires_grad=True)
        tangent_head = JacModule.from_linear(ltan)

        phi_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, phi_head]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_head_net_siren_deeper(self) -> None:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        DATA_SCALE = 3.
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        l1 = torch.nn.Linear(3 + 9, H)
        torch.nn.init.kaiming_uniform_(l1.weight, mode='fan_in', nonlinearity='relu')
        l1.weight = torch.nn.Parameter(DATA_SCALE * l1.weight, requires_grad=True)
        module1 = JacModule.from_linear(l1)
        #module2 = JacModule.from_relu(torch.nn.ReLU())
        #module2 = JacModule.from_tanh(torch.nn.Tanh())
        module2 = JacModule.from_sin()
        l2 = torch.nn.Linear(H, H)
        torch.nn.init.kaiming_uniform_(l2.weight, mode='fan_in', nonlinearity='relu')
        #l2.weight = torch.nn.Parameter(math.sqrt(3) * l2.weight, requires_grad=True)
        module3 = JacModule.from_linear(l2)
        #module4 = JacModule.from_relu(torch.nn.ReLU())
        #module4 = JacModule.from_tanh(torch.nn.Tanh())
        module4 = JacModule.from_sin()

        l41 = torch.nn.Linear(H, H)
        torch.nn.init.kaiming_uniform_(l41.weight, mode='fan_in', nonlinearity='relu')
        #l2.weight = torch.nn.Parameter(math.sqrt(3) * l2.weight, requires_grad=True)
        module41 = JacModule.from_linear(l41)
        #module4 = JacModule.from_relu(torch.nn.ReLU())
        #module4 = JacModule.from_tanh(torch.nn.Tanh())
        module42 = JacModule.from_sin()

        lphi = torch.nn.Linear(H, self.k())#, bias=False)
        torch.nn.init.kaiming_uniform_(lphi.weight, mode='fan_in', nonlinearity='relu')
        #lphi.weight = torch.nn.Parameter(math.sqrt(3) * lphi.weight,requires_grad=True)
        phi_head = JacModule.from_linear(lphi)

        ltan = torch.nn.Linear(H, 2 * self.k(), bias=False)
        torch.nn.init.kaiming_uniform_(ltan.weight, mode='fan_in', nonlinearity='relu')
        #ltan.weight = torch.nn.Parameter(math.sqrt(3) * ltan.weight,requires_grad=True)
        tangent_head = JacModule.from_linear(ltan)

        phi_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, module41, module42, phi_head]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, module41, module42, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_head_net_thin(self) -> None:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        module1 = JacModule.from_linear(torch.nn.Linear(3 + 9, H))
        #module2 = JacModule.from_relu(torch.nn.ReLU())
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        #module2 = JacModule.from_sin()
        phi_head = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=False))
        tangent_head = JacModule.from_linear(torch.nn.Linear(H, 2 * self.k(), bias=False))

        phi_modules = torch.nn.ModuleList([module0, module1, module2, phi_head]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([module0, module1, module2, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_head_net_thin_siren(self) -> None:
        H = self.H
        #state_aug = StateAugment2D(torch.tensor([0, math.pi/2]))
        #state_aug_module = JacModule.from_jac_enabled(state_aug)
        DATA_SCALE = 1.
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        l1 = torch.nn.Linear(3 + 9, H)
        torch.nn.init.kaiming_uniform_(l1.weight, mode='fan_in')
        #l1.weight = torch.nn.Parameter(DATA_SCALE * math.sqrt(3) * l1.weight, requires_grad=True)
        module1 = JacModule.from_linear(l1)

        module2 = JacModule.from_sin()

        lphi = torch.nn.Linear(H, self.k())
        torch.nn.init.kaiming_uniform_(lphi.weight, mode='fan_in')
        #lphi.weight = torch.nn.Parameter(math.sqrt(3) * lphi.weight,requires_grad=True)
        phi_head = JacModule.from_linear(lphi)

        ltan = torch.nn.Linear(H, 2 * self.k(), bias=False)
        torch.nn.init.kaiming_uniform_(ltan.weight, mode='fan_in')
        #ltan.weight = torch.nn.Parameter(math.sqrt(3) * ltan.weight,requires_grad=True)
        tangent_head = JacModule.from_linear(ltan)

        phi_modules = torch.nn.ModuleList([module0, module1, module2, phi_head]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))

        tangent_modules = torch.nn.ModuleList([module0, module1, module2, tangent_head]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_linear(self) -> None:
        
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        if self.learn_normal:
            phi_head = JacModule.from_linear(torch.nn.Linear(3 + 9, self.k(), bias=False))
            

            phi_modules = torch.nn.ModuleList([module0, phi_head]) 

            self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))
        if self.learn_tangent:
            tangent_head = JacModule.from_linear(torch.nn.Linear(3 + 9, 2 * self.k(), bias=False))
            tangent_modules = torch.nn.ModuleList([module0, tangent_head]) 
            self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_phi_net_sum(self) -> None:
        module1 = QuadSum3D(self.k())

        self.phi_net = JacNet(JacModule.from_jac_enabled(module1))

    def init_phi_net_rotinv(self) -> None:
        H = self.H
        module1 = JacModule.from_linear(torch.nn.Linear(7, H))
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        module5 = JacModule.from_linear(torch.nn.Linear(H, self.k()*3, bias=True))
        vertex_modules = torch.nn.ModuleList([module1, module2, module3, module4, module5]) 
        vertex_net = JacModule.from_jac_enabled(Series(vertex_modules))

        vertex_augment = JacModule.from_jac_enabled(VertexAugment3D(vertex_net))

        zvec = torch.tensor([[0.0, 0, 1]])
        vertex_dist_extract = JacModule.from_jac_enabled(VertexDistExtract3D(zvec))

        modules = torch.nn.ModuleList([vertex_augment, vertex_dist_extract]) 
        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(modules)))
        

    def init_phi_net_deep(self) -> None:
        H = self.H
        module1 = JacModule.from_linear(torch.nn.Linear(7, H))
        #module1d = JacModule.from_dropout(torch.nn.Dropout(0.2))
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        #module3d = JacModule.from_dropout(torch.nn.Dropout(0.2))
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        module5 = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=True))

        #modules = torch.nn.ModuleList([module1, module1d, module2, module3, module3d, module4, module5]) 
        modules = torch.nn.ModuleList([module1, module2, module3, module4, module5]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_phi_net_deep_quad(self) -> None:
        H = 128
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        module1 = JacModule.from_linear(torch.nn.Linear(3 + 9, H))
        #module1d = JacModule.from_dropout(torch.nn.Dropout(0.2))
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        #module3d = JacModule.from_dropout(torch.nn.Dropout(0.2))
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        module5 = JacModule.from_linear(torch.nn.Linear(H, self.k(), bias=False))

        modules = torch.nn.ModuleList([module0, module1, module2, module3, module4, module5]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_phi_net_quad(self) -> None:
        module0 = JacModule.from_quad()
        module1 = JacModule.from_linear(torch.nn.Linear(7 + 7*7, self.k(), bias=True))

        modules = torch.nn.ModuleList([module0, module1]) 

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(modules)))

    def init_tangent_net_deep(self) -> None:
        H = self.H
        module1 = JacModule.from_linear(torch.nn.Linear(7, H))
        module2 = JacModule.from_tanh(torch.nn.Tanh())
        module3 = JacModule.from_linear(torch.nn.Linear(H, H))
        module4 = JacModule.from_tanh(torch.nn.Tanh())
        module5 = JacModule.from_linear(torch.nn.Linear(H, 2 * self.k(), bias=True))

        modules = torch.nn.ModuleList([module1, module2, module3, module4, module5]) 

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(modules)))


    def init_vertex(self) -> None:
        vertices = torch.nn.Parameter(1.0 * torch.tensor([[1, -1, 1],
                                                    [1, 1, 1],
                                                    [-1, 1, 1],
                                                    [-1, -1, 1],
                                                    [1, -1, -1],
                                                    [1, 1, -1],
                                                    [-1, 1, -1],
                                                    [-1, -1, -1]]).double().t(),
                                                    requires_grad=True)
        
        module0 = JacModule.from_jac_enabled(StateAugment3D())
        #pdb.set_trace()
        (wn, wt) = utils.vertices_to_pR_weights(self.geometry.vertices)
        if self.learn_normal:
            lphi = torch.nn.Linear(3 + 9, self.k(), bias=False)
            lphi.weight = torch.nn.Parameter(wn, requires_grad=True)
            with torch.no_grad():
                lphi.weight.add_(torch.randn(lphi.weight.size()) * 0.01)
            phi_head = JacModule.from_linear(lphi)
            

            phi_modules = torch.nn.ModuleList([module0, phi_head]) 

            self.phi_net = JacNet(JacModule.from_jac_enabled(Series(phi_modules)))
        if self.learn_tangent:
            ltan = torch.nn.Linear(3 + 9, 2 * self.k(), bias=False)
            ltan.weight = torch.nn.Parameter(self.mu * wt, requires_grad=True)
            with torch.no_grad():
                ltan.weight.add_(torch.randn(ltan.weight.size()) * 0.01)
            tangent_head = JacModule.from_linear(ltan)
            tangent_modules = torch.nn.ModuleList([module0, tangent_head]) 
            self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(tangent_modules)))

    def init_phi_net(self) -> None:
        vertices = torch.nn.Parameter(torch.tensor([[1, -1, 1],
                                                    [1, 1, 1],
                                                    [-1, 1, 1],
                                                    [-1, -1, 1],
                                                    [1, -1, -1],
                                                    [1, 1, -1],
                                                    [-1, 1, -1],
                                                    [-1, -1, -1]]).double().t(),
                                                    requires_grad=True)
        zvec = torch.tensor([[0.0, 0, 1]])

        # with torch.no_grad():
            # vertices.add_(torch.randn(vertices.size()) * 0.3)

        module1 = DistExtract3D(zvec, vertices)

        self.phi_net = JacNet(JacModule.from_jac_enabled(Series(torch.nn.ModuleList([JacModule.from_jac_enabled(module1)]))))

    def init_tangent_net(self) -> None:
        #MU_START = 1.0
        vertices = torch.nn.Parameter(torch.tensor([[1, -1, 1],
                                                    [1, 1, 1],
                                                    [-1, 1, 1],
                                                    [-1, -1, 1],
                                                    [1, -1, -1],
                                                    [1, 1, -1],
                                                    [-1, 1, -1],
                                                    [-1, -1, -1]]).double().t(),
                                                    requires_grad=True)
        xvec = torch.tensor([[1.0, 0, 0]])
        yvec = torch.tensor([[0.0, 1, 0]])

        # with torch.no_grad():
            # vertices.add_(torch.randn(vertices.size()) * 0.3)

        module1 = DistExtract3D(self.mu * torch.cat((xvec, yvec), dim=0), vertices)

        self.tangent_net = JacNet(JacModule.from_jac_enabled(Series(torch.nn.ModuleList([JacModule.from_jac_enabled(module1)]))))

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
