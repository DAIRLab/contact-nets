import torch
from torch import Tensor
from torch.nn import Module

from contactnets.system import SystemParams
from contactnets.entity import Dynamic2D, Dynamic2DParams

import unittest
import torchtestcase as ttc

from dataclasses import dataclass

import math

import pdb

torch.set_default_tensor_type(torch.DoubleTensor)
batch_n = 2

class TestDynamic2D(ttc.TorchTestCase):
    def setUp(self):
        configuration = Tensor([0, 1, 0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 
        velocity = Tensor([0, 1, 0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 
        params = Dynamic2DParams(mass = torch.tensor(2.0),
                                 inertia = torch.tensor(3.0))
        
        self.entity = Dynamic2D(configuration, velocity, params)
        self.entity.append_control(Tensor([1, 0, 0]).reshape(1, 3, 1).repeat(batch_n, 1, 1))

        self.sp = SystemParams(torch.tensor(0.1), torch.tensor(3.0))

    def test_f(self):
        f = self.entity.compute_f_history(self.sp)
        self.assertEqual(f, Tensor([0.05, 0.7, 0]).reshape(1, 3, 1).repeat(2, 1, 1))

