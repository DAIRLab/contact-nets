import torch
from torch import Tensor
from torch.nn import Module

from contactnets.system import SystemParams
from contactnets.entity import Dynamic2D, Dynamic2DParams, Ground2D
from contactnets.interaction import PolyGround2D, LCP

import unittest
import torchtestcase as ttc

from dataclasses import dataclass

import math

import pdb

torch.set_default_tensor_type(torch.DoubleTensor)
batch_n = 2

class TestLCP2D(ttc.TorchTestCase):
    def setUp(self):
        self.dynamic = self.make_dynamic()
        self.dynamic.append_control(Tensor([1, 0, 0]).reshape(1, 3, 1).repeat(batch_n, 1, 1))
        self.ground = Ground2D(batch_n, torch.tensor(0.0))

        vertices = torch.tensor([[1.0, 1.0, -1.0, -1.0],
                                 [-1.0, 1.0, 1.0, -1.0]])

        self.poly_ground_2d = PolyGround2D(self.dynamic, self.ground, vertices, torch.tensor(1.0))

        G_bases = torch.tensor([1.0, -1.0])

        self.lcp = LCP(self.poly_ground_2d, G_bases)

        self.sp = SystemParams(torch.tensor(0.1), torch.tensor(3.0))

    def make_dynamic(self):
        configuration = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 
        velocity = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 

        params = Dynamic2DParams(mass = torch.tensor(2.0),
                                 inertia = torch.tensor(3.0))
        
        return Dynamic2D(configuration, velocity, params)

    def test_phi(self):
        phi = self.poly_ground_2d.compute_phi_history() 
        self.assertEqual(phi, torch.tensor([0.0, 2, 2, 0]).reshape(1, 4, 1).repeat(batch_n, 1, 1))
    
    def test_Jn(self):
        Jn = self.poly_ground_2d.compute_Jn_history()
        Jn_true = torch.cat((torch.zeros(4, 1), torch.ones(4, 1), torch.tensor([[1.0],[1],[-1],[-1]])), dim=1)
        Jn_true_batch = Jn_true.unsqueeze(0).repeat(batch_n, 1, 1)
        self.eps = 1e-4
        self.assertEqual(Jn, Jn_true_batch)

    def test_Jt_tilde(self):
        Jt_tilde = self.poly_ground_2d.compute_Jt_tilde_history()
        Jt_tilde_true = torch.cat((torch.ones(4, 1), torch.zeros(4, 1), torch.tensor([[1.0],[-1],[-1],[1]])), dim=1)
        Jt_tilde_true_batch = Jt_tilde_true.unsqueeze(0).repeat(batch_n, 1, 1)
        self.eps = 1e-4
        self.assertEqual(Jt_tilde, Jt_tilde_true_batch)
    
    def test_M(self):
        M = self.poly_ground_2d.compute_M_history()
        M_true = 2 * torch.eye(3).unsqueeze(0).repeat(batch_n, 1, 1)
        self.assertEqual(M, M_true)

    def test_M_i(self):
        M_i = self.poly_ground_2d.compute_M_i_history()
        M_i_true = 0.5 * torch.eye(3).unsqueeze(0).repeat(batch_n, 1, 1)
        self.assertEqual(M_i, M_i_true)

    def test_gamma(self):
        gamma = self.poly_ground_2d.compute_gamma_history()
        gamma_true = torch.eye(3).unsqueeze(0).repeat(batch_n, 1, 1)
        self.assertEqual(gamma, gamma_true)

    def test_Jt(self):
        Jt = self.lcp.compute_Jt()
        Jt_tilde_true = torch.cat((torch.ones(4, 1), torch.zeros(4, 1), torch.tensor([[1.0],[-1],[-1],[1]])), dim=1)

        Jt_true = Jt_tilde_true.repeat(1, 2)
        Jt_true[:, 3:6] *= -1
        Jt_true = Jt_true.reshape(-1, 3)
        Jt_true_batch = Jt_true.unsqueeze(0).repeat(batch_n, 1, 1)

        self.eps = 1e-4
        self.assertEqual(Jt, Jt_true_batch)

    def test_impulses(self):
        self.dynamic.clear_history()
        self.dynamic.clear_interactions()

        configuration = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 
        velocity = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1).repeat(batch_n, 1, 1) 

        self.dynamic.set_state(configuration, velocity)
        self.dynamic.append_control(Tensor([0, 0, 0]).reshape(1, 3, 1).repeat(batch_n, 1, 1))

        impulses = self.lcp.compute_impulses(self.sp)

        impulses_true = torch.zeros(2, 8, 1)
        impulses_true[:, 0, :] = 0.3
        impulses_true[:, 3, :] = 0.3

        self.eps = 1e-4
        self.assertEqual(impulses, impulses_true)
