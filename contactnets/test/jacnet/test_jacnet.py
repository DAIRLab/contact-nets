import torch

from contactnets.jacnet import JacNet
from contactnets.jacnet import JacModule
from contactnets.jacnet import Series

import torchtestcase as ttc

import math

import unittest

import pdb

torch.set_default_tensor_type(torch.DoubleTensor)

class TestJacNet(ttc.TorchTestCase):
    def test_linear_relu(self):
        activation1 = torch.nn.ReLU()

        operation1 = torch.nn.Linear(1, 2, bias=True)
        operation1.weight = torch.nn.Parameter(torch.tensor([[-1.0], [1]]))
        operation1.bias = torch.nn.Parameter(torch.tensor([0.0, 1]))

        activation2 = torch.nn.ReLU()

        operation2 = torch.nn.Linear(2, 3, bias=True)
        operation2.weight = torch.nn.Parameter(torch.tensor([[-1.0, 2], [0, 3], [1, 3]]))
        operation2.bias = torch.nn.Parameter(torch.tensor([2.0, 2, 2]))

        modules = torch.nn.ModuleList([JacModule.from_relu(activation1),
                                       JacModule.from_linear(operation1),
                                       JacModule.from_relu(activation2),
                                       JacModule.from_linear(operation2)])

        net = JacNet(JacModule.from_jac_enabled(Series(modules)))

        x = torch.tensor([[[1.0]]])

        self.assertEqual(net(x), torch.tensor([[[6.0, 8, 8]]]))
        self.assertEqual(net.jacobian(x), torch.tensor([[[2.0, 3, 3]]]))

if __name__ == '__main__': unittest.main()
