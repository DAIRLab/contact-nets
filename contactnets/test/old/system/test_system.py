import torch
from torch import Tensor
from torch.nn import Module

from contactnets.system import System, SystemParams
from contactnets.entity import Dynamic2D, Dynamic2DParams, Ground2D
from contactnets.interaction import PolyGround2D, LCP
from contactnets.utils import system_io 

import unittest
import torchtestcase as ttc

from dataclasses import dataclass

import math

import pdb

torch.set_default_tensor_type(torch.DoubleTensor)
batch_n = 2
step_n = 1

class TestSystem(ttc.TorchTestCase):
    def setUp(self):
        self.dynamic = self.make_dynamic()
        self.dynamic.append_control(Tensor([1, 0, 0]).reshape(1, 3, 1).repeat(batch_n, 1, step_n))

        self.ground = Ground2D(batch_n, torch.tensor(0.0))
        self.ground.append_control(torch.zeros(batch_n, 0, step_n))

        vertices = torch.tensor([[1.0, 1.0, -1.0, -1.0],
                                 [-1.0, 1.0, 1.0, -1.0]])

        self.poly_ground_2d = PolyGround2D(self.dynamic, self.ground, vertices, torch.tensor(1.0))

        G_bases = torch.tensor([1.0, -1.0])


        self.lcp = LCP(self.poly_ground_2d, G_bases)

        self.sp = SystemParams(torch.tensor(0.1), torch.tensor(3.0))


        self.dynamic2 = self.make_dynamic()
        self.dynamic2.append_control(Tensor([2, 0, 0]).reshape(1, 3, 1).repeat(batch_n, 1, step_n))

        self.system = System([self.dynamic, self.dynamic2, self.ground], [self.lcp], self.sp)

    def make_dynamic(self):
        configuration = Tensor([0, 1, 0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 
        velocity = Tensor([0, 1, 0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 

        params = Dynamic2DParams(mass = torch.tensor(2.0),
                                 inertia = torch.tensor(3.0))
        
        return Dynamic2D(configuration, velocity, params)

    def test_serialize(self):
        x = system_io.serialize_system(self.system)
        x_true = Tensor([0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0]).reshape(1, 1, 18).repeat(2, 1, 1)
        self.assertEqual(x, x_true)

    def test_load(self):
        x = Tensor([0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0]).reshape(1, 1, 18).repeat(2, 1, 1)
        system_io.load_system(x, self.system)

        self.assertEqual(len(self.dynamic.configuration_history), step_n)
        self.assertEqual(self.dynamic.configuration_history[0], Tensor([[[0], [1], [0]], [[0], [1], [0]]]))
