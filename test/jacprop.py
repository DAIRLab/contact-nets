import pdb  # noqa
import unittest

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torchtestcase as ttc

from contactnets.jacprop import (BiasSum, JacProp, ModuleAugment, ParallelSum, Scale,
                                 StateAugment2D, StateAugment3D, TransformAndProject3D,
                                 TransformAndProjectInput3D)

import torch.autograd.functional as auto_F


class TestJacProp(ttc.TorchTestCase):
    def setUp(self):
        self.jacprop = JacProp()
        torch.set_grad_enabled(True)
        self.eps = 1e-6

    def assert_same_jacobian(self, module: Module, input: Tensor):
        self.jacprop.jacobian_enable(module)
        jac_jacprop = self.jacprop.jacobian(module, input).squeeze(0)
        jac_auto = auto_F.jacobian(module, input, create_graph=True).squeeze(0).squeeze(1)

        try:
            self.assertEqual(jac_jacprop, jac_auto)
        except Exception:
            print('\n-------------------------------------------')
            print('Jac prop jacobian:')
            print(jac_jacprop)
            print('Auto_F jacobian:')
            print(jac_auto)

            if jac_jacprop.shape == jac_auto.shape:
                print('Difference:')
                print(jac_jacprop - jac_auto)

            raise


    ################################################################################
    #                              Built in functions                              #
    ################################################################################


    def test_linear(self):
        self.assert_same_jacobian(nn.Linear(5, 3), torch.rand(1, 5))

    def test_relu(self):
        self.assert_same_jacobian(nn.ReLU(), torch.rand(1, 5))

    def test_tanh(self):
        self.assert_same_jacobian(nn.Tanh(), torch.rand(1, 5))

    def test_sequential(self):
        module = nn.Sequential(
            nn.Linear(5, 3),
            nn.Tanh(),
            nn.Linear(3, 20),
            nn.ReLU()
        )
        self.assert_same_jacobian(module, torch.rand(1, 5))


    ################################################################################
    #                             Additional functions                             #
    ################################################################################


    def test_bias_sum(self):
        module = BiasSum(nn.Linear(5, 3), nn.Parameter(torch.tensor(3.0)))
        self.assert_same_jacobian(module, torch.rand(1, 5))

    def test_parallel_sum(self):
        module = ParallelSum(nn.Linear(5, 3), nn.Linear(5, 3))
        self.assert_same_jacobian(module, torch.rand(1, 5))

    def test_module_augment(self):
        module = ModuleAugment(nn.Linear(5, 3))
        self.assert_same_jacobian(module, torch.rand(1, 5))

    def test_scale(self):
        module = Scale(torch.tensor(3.0))
        self.assert_same_jacobian(module, torch.rand(1, 5))

    def test_state_augment_2d(self):
        module = StateAugment2D(torch.tensor([3.0, 4.0]))
        self.assert_same_jacobian(module, torch.rand(1, 3))

    def test_state_augment_3d(self):
        module = StateAugment3D()
        self.assert_same_jacobian(module, torch.rand(1, 7))

    def test_transform_and_project_3d(self):
        projections = torch.rand(2, 3)
        points = torch.rand(4, 3)
        module = TransformAndProject3D(projections, points)
        self.assert_same_jacobian(module, torch.rand(1, 7))

    def test_transform_and_project_input_3d(self):
        projections = torch.rand(2, 3)
        module = TransformAndProjectInput3D(projections)
        input = torch.rand(1, 10)
        input[:, 3:7] /= torch.norm(input[:, 3:7])  # normalize quaternion
        self.assert_same_jacobian(module, input)


if __name__ == '__main__': unittest.main()
