import torch

from contactnets.jacnet import JacModule
from contactnets.jacnet import Scale
from contactnets.jacnet import ParallelSum
from contactnets.jacnet import Series
from contactnets.jacnet import StateAugment2D
from contactnets.jacnet import VertexAugment3D
from contactnets.jacnet import VertexDistExtract3D
from contactnets.jacnet import DistExtract3D

import torchtestcase as ttc

import math

import unittest

import pdb

torch.set_default_tensor_type(torch.DoubleTensor)

class TestJacModule(ttc.TorchTestCase):
    def test_linear(self):
        linear = torch.nn.Linear(2, 3)
        linear.weight.data.fill_(5.0)
        linear.bias.data.fill_(0.0)

        linear_jm = JacModule.from_linear(linear)

        val = linear_jm(torch.ones(1, 1, 2))
        jac = linear_jm.jacobian(torch.eye(3).unsqueeze(0))
        
        self.assertEqual(val, torch.tensor([[[10.0, 10.0, 10.0]]]))
        self.assertEqual(jac, 5.0 * torch.ones(1, 2,3))

    def test_relu_inactive(self):
        relu = torch.nn.ReLU()

        relu_jm = JacModule.from_relu(relu)
        
        val = relu_jm(torch.ones(1, 1, 3))
        jac = relu_jm.jacobian(torch.eye(3).unsqueeze(0))
        
        self.assertEqual(val, torch.ones(1, 1, 3))
        self.assertEqual(jac, torch.eye(3).unsqueeze(0))

    def test_relu_active(self):
        relu = torch.nn.ReLU()

        relu_jm = JacModule.from_relu(relu)

        val = relu_jm(-torch.ones(1, 1, 3))
        jac = relu_jm.jacobian(torch.eye(3).unsqueeze(0))

        self.assertEqual(val, torch.zeros(1, 1, 3))
        self.assertEqual(jac, torch.zeros(1, 3, 3))

    def test_relu_mixed(self):
        relu = torch.nn.ReLU()

        relu_jm = JacModule.from_relu(relu)

        val = relu_jm(torch.tensor([1.0, -1.0, 1.0]).reshape(1, 1, 3))
        jac = relu_jm.jacobian(torch.eye(3).unsqueeze(0))

        self.assertEqual(val, torch.tensor([[[1.0, 0.0, 1.0]]]))
        self.assertEqual(jac, torch.diagflat(torch.tensor([1.0, 0.0, 1.0])).unsqueeze(0))

    def test_tanh(self):
        tanh = torch.nn.Tanh()

        tanh_jm = JacModule.from_tanh(tanh)
        
        val = tanh_jm(torch.tensor([-2.0, 0, 2]).reshape(1, 1, 3))
        jac = tanh_jm.jacobian(torch.eye(3).unsqueeze(0))

        self.eps = 1e-4
        self.assertEqual(jac, torch.diagflat(torch.tensor([math.sqrt(2) / 20, 1.0, math.sqrt(2) / 20])).unsqueeze(0))

    def test_parallel_sum(self):
        linear1 = torch.nn.Linear(1, 2)
        linear1.weight.data.fill_(5.0)
        linear1.bias.data.fill_(0.0)
        linear1_jm = JacModule.from_linear(linear1)

        linear2 = torch.nn.Linear(1, 2)
        linear2.weight = torch.nn.Parameter(torch.tensor([[2.0], [3]]))
        linear2.bias.data.fill_(0.0)
        linear2_jm = JacModule.from_linear(linear2)

        modules = torch.nn.ModuleList([linear1_jm, linear2_jm])

        ps_jm = JacModule.from_jac_enabled(ParallelSum(modules))

        val = ps_jm(torch.tensor([2.0]).reshape(1, 1, 1))
        jac = ps_jm.jacobian(torch.eye(2).unsqueeze(0))

        self.assertEqual(val, torch.tensor([14.0, 16]).reshape(1, 1, 2))
        self.assertEqual(jac, torch.tensor([7.0, 8]).reshape(1, 1, 2))

    def test_series(self):
        linear = torch.nn.Linear(2, 3)
        linear.weight.data.fill_(5.0)
        linear.bias.data.fill_(0.0)

        linear_jm = JacModule.from_linear(linear)

        relu_jm = JacModule.from_relu(torch.nn.ReLU())

        modules = torch.nn.ModuleList([linear_jm, relu_jm])
        series_jm = JacModule.from_jac_enabled(Series(modules))

        val = series_jm(torch.ones(1, 1, 2))
        jac = series_jm.jacobian(torch.eye(3).unsqueeze(0))
        
        self.assertEqual(val, torch.tensor([10.0, 10.0, 10.0]).reshape(1, 1, 3))
        self.assertEqual(jac, 5.0 * torch.ones((1, 2,3)))

    def test_scale(self):
        scalar = torch.tensor(2.0)
        
        scale_jm = JacModule.from_jac_enabled(Scale(scalar))

        val = scale_jm(torch.tensor([1.0, 2.0]).reshape(1, 1, 2))
        jac = scale_jm.jacobian(torch.eye(2).unsqueeze(0))

        self.assertEqual(val, torch.tensor([2.0, 4.0]).reshape(1, 1, 2))
        self.assertEqual(jac, 2.0 * torch.eye(2).unsqueeze(0))

    def test_state_augment_2d(self):
        phases = torch.tensor([0.0, math.pi / 2])

        sa_jm = JacModule.from_jac_enabled(StateAugment2D(phases))

        val = sa_jm(torch.tensor([1.0, 1.0, 0.0]).reshape(1, 1, 3))
        jac = sa_jm.jacobian(torch.eye(4).unsqueeze(0)) 

        correct_val = torch.tensor([1.0, 1.0, 0.0, 1.0]).reshape(1, 1, 4) 
        
        eye_sliced = torch.eye(3).unsqueeze(0)[:, :, 0:2]
        augment_jac = torch.tensor([[0.0, 0], [0, 0], [1, 0]]).unsqueeze(0)
        correct_jac = torch.cat((eye_sliced, augment_jac), dim=2)
        
        self.eps = 1e-6
        self.assertEqual(val, correct_val)
        self.assertEqual(jac, correct_jac)

    def test_vertex_augment_3d(self):
        linear = torch.nn.Linear(7, 3)
        linear.weight.data.fill_(1.0)
        linear.bias.data.fill_(0.0)
        vertex_jm = JacModule.from_linear(linear)

        va_jm = JacModule.from_jac_enabled(VertexAugment3D(vertex_jm))
        
        q = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 1, 7) 
        val = va_jm(q)
        jac = va_jm.jacobian(torch.eye(10).unsqueeze(0)) 
        
        correct_val = torch.cat((q, torch.tensor([2.0, 2, 2]).reshape(1, 1, 3)), dim=2) 
        correct_jac = torch.cat((torch.eye(7), linear.weight), dim=0).t().unsqueeze(0)
        
        self.eps = 1e-6
        self.assertEqual(val, correct_val)
        self.assertEqual(jac, correct_jac)

    def test_vertex_dist_extract_3d(self):
        q_vert = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2, 2, 2]).reshape(1, 1, 10) 
        zvec = torch.tensor([[0.0, 0, 1]])

        vd_jm = JacModule.from_jac_enabled(VertexDistExtract3D(zvec))
        val = vd_jm(q_vert)
        jac = vd_jm.jacobian(torch.eye(1).unsqueeze(0))

        correct_val = torch.tensor([[[2.0]]])
        correct_jac = torch.tensor([0.0, 0, 1, 4, 4, -4, 0, 0, 0, 1]).reshape(1, 10, 1)

        self.eps = 1e-6
        self.assertEqual(val, correct_val)
        self.assertEqual(jac, correct_jac)

if __name__ == '__main__': unittest.main()
