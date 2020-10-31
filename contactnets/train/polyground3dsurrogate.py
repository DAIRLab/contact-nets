import torch
from torch import Tensor
from torch.nn import Module

from abc import ABC, abstractmethod
from typing import *

from contactnets.interaction import Interaction, PolyGround3D, LCP
from contactnets.system import System
from contactnets.train import Loss, LossType

from contactnets.utils import utils, solver_utils, tensor_utils

import math
import numpy as np

from dataclasses import dataclass

import pdb

@dataclass
class SurrogateConfig3D:
    w_comp_n:               float = 0.0
    w_comp_t:               float = 0.0
    w_match:                float = 0.0
    w_cone:                 float = 0.0
    w_penetration_slack:    float = 0.0

    w_penetration:          float = 0.0
    w_config_grad_normal:   float = 0.0
    w_config_grad_tangent:  float = 0.0
    w_config_grad_perp:     float = 0.0
    w_st_estimate_pen:      float = 0.0
    w_st_estimate_normal:   float = 0.0
    w_st_estimate_tangent:  float = 0.0
    w_tangent_jac_d2:       float = 0.0

    w_contact_threshold:    float = -1.0

class PolyGround3DSurrogate(Loss):
    interaction: Interaction
    G_bases: Tensor
    penalize_penetration: bool
    constrain_config_grad_normal: bool
    constrain_config_grad_tangent: bool
    constrain_st_estimate_normal: bool
    constrain_st_estimate_tangent: bool

    def __init__(self, system: System, interaction: PolyGround3D,
                       G_bases: Tensor = None, config = SurrogateConfig3D()) -> None:
        super(PolyGround3DSurrogate, self).__init__(
                ['surrogate', 'surr_qp',
                 'surr_qp_comp_n', 'surr_qp_comp_t', 'surr_qp_match',
                 'surr_qp_cone', 'surr_qp_pen_slack',
                 'surr_reg_pen', 'surr_reg_grad_normal',
                 'surr_reg_grad_tangent', 'surr_reg_grad_perp', 'surr_reg_st_pen',
                 'surr_reg_st_normal', 'surr_reg_st_tangent', 'surr_reg_tangent_d2'],
                system, LossType.STEPWISE, False)

        self.interaction = interaction

        if G_bases is None:
            # Can't be in argument b/c messes default tensor type
            G_bases = self.compute_G_bases(4)
        self.G_bases = G_bases

        self.config = config

    def compute_loss(self, meta_save_file: str = None) -> Tensor:
        sp = self.system.params
        config = self.config

        # Get configuration / velocity for polygon
        configuration = self.interaction.poly.get_configuration_previous()
        configuration_plus = self.interaction.poly.get_configuration()
        velocity = self.interaction.poly.get_velocity_previous()
        velocity_plus = self.interaction.poly.get_velocity()
        control = self.interaction.poly.get_control_previous()

        batch_n = self.interaction.batch_n()
        bases_n = self.G_bases.shape[0]
        k = self.interaction.k()

        G = self.compute_G()

        phi = self.interaction.compute_phi_previous()
        phi_plus = self.interaction.compute_phi_history()

        Jn = self.interaction.compute_Jn_previous()
        Jt_tilde = self.interaction.compute_Jt_tilde_previous()
        Jt = G.transpose(1,2).bmm(Jt_tilde)
        J = torch.cat((Jn, Jt), dim=1)
        J_tilde = torch.cat((Jn, Jt_tilde), dim=1)

        E_2 = tensor_utils.veceye(k, 2).unsqueeze(0).repeat(batch_n, 1, 1)

        gamma = self.interaction.compute_gamma_previous()
        f = self.interaction.poly.compute_f_previous(sp)

        M = self.interaction.compute_M_previous()
        M_i = self.interaction.compute_M_i_previous()

        control_dv = M_i.bmm(control) * sp.dt
        gravity_dv = sp.g * Tensor([0, 0, -1, 0, 0, 0]).reshape(1, 6, 1).repeat(batch_n, 1, 1) * sp.dt

        v_diff = velocity_plus - velocity

        F_data = M.bmm(v_diff - gravity_dv - control)


        # Optimization variables are lambda_n, lambda_t
        normal_mat_pad = lambda x: tensor_utils.diag_prepend_mat(x, 0, k)
        normal_vec_pad = lambda x: tensor_utils.pad_mat_left(x, 0, k)

        tangent_mat_pad = lambda x: tensor_utils.diag_expand_mat(x, 0, 2 * k)
        tangent_vec_pad = lambda x: tensor_utils.pad_mat_right(x, 0, 2 * k)

        slack_mat_pad = lambda x: tensor_utils.diag_expand_mat(x, 0, k)
        slack_vec_pad = lambda x: tensor_utils.pad_mat_right(x, 0, k)

        # lambda_n and phi complementarity
        comp_n_A = config.w_comp_n * tensor_utils.matrix_diag((phi_plus.squeeze(2) ** 2))
        comp_n_b = config.w_comp_n * torch.zeros(batch_n, 1, k)
        comp_n_c = config.w_comp_n * torch.zeros(batch_n, 1, 1)

        comp_n_A = slack_mat_pad(tangent_mat_pad(comp_n_A))
        comp_n_b = slack_vec_pad(tangent_vec_pad(comp_n_b))

        # lambda_t and phi complementarity
        phi_expand = E_2.bmm(phi_plus)
        comp_t_A = config.w_comp_t * tensor_utils.matrix_diag((phi_expand.squeeze(2) ** 2))
        comp_t_b = config.w_comp_t * torch.zeros(batch_n, 1, 2 * k)
        comp_t_c = config.w_comp_t * torch.zeros(batch_n, 1, 1)

        comp_t_A = slack_mat_pad(normal_mat_pad(comp_t_A))
        comp_t_b = slack_vec_pad(normal_vec_pad(comp_t_b))

        # Match impulse data (multiply by M_i to get scaling)
        # Term of form (M_i gamma^T [Jn, Jt_tilde]^T lambda - M_i F)^2
        match_quad_A = M_i.bmm(gamma.transpose(1,2)).bmm(J_tilde.transpose(1,2))
        match_quad_b = M_i.bmm(F_data)
        match_A = config.w_match * match_quad_A.transpose(1, 2).bmm(match_quad_A)
        match_b = config.w_match * (-2) * match_quad_b.transpose(1, 2).bmm(match_quad_A)
        match_c = config.w_match * match_quad_b.transpose(1, 2).bmm(match_quad_b)

        match_A = slack_mat_pad(match_A)
        match_b = slack_vec_pad(match_b)

        # Friction cone boundary
        sliding_vels = Jt_tilde.bmm(gamma).bmm(velocity_plus)
        #pdb.set_trace()
        #cone_normal_mat = torch.diagflat(sliding_vels).unsqueeze(0).bmm(E_2)
        cone_normal_mat = tensor_utils.matrix_diag(sliding_vels.squeeze(2)).bmm(E_2)
        
        sliding_vel_norms = E_2.transpose(1,2).bmm(sliding_vels.mul(sliding_vels))
        sliding_vel_norms = E_2.bmm(torch.sqrt(sliding_vel_norms))
        #cone_tangent_mat = torch.diagflat(sliding_vel_norms).unsqueeze(0)
        cone_tangent_mat = tensor_utils.matrix_diag(sliding_vel_norms.squeeze(2))
        cone_mat = torch.cat((cone_normal_mat, cone_tangent_mat), dim=2)

        cone_A = config.w_cone * cone_mat.transpose(1,2).bmm(cone_mat)
        cone_b = config.w_cone * torch.zeros(batch_n, 1, 3 * k)
        cone_c = config.w_cone * torch.zeros(batch_n, 1, 1)

        cone_A = slack_mat_pad(cone_A)
        cone_b = slack_vec_pad(cone_b)

        # Penalize slack variables (penetration phi)
        slack_A = config.w_penetration_slack * tensor_utils.diag_expand_mat(torch.zeros(k * (2+1), k * (2+1)).unsqueeze(0).repeat(batch_n, 1, 1), 1.0, k)
        slack_b = config.w_penetration_slack * torch.zeros(batch_n, 1, 4 * k)
        slack_c = config.w_penetration_slack * torch.zeros(batch_n, 1, 1)

        A = comp_n_A + comp_t_A + match_A + cone_A + slack_A
        b = comp_n_b + comp_t_b + match_b + cone_b + slack_b
        c = comp_n_c + comp_t_c + match_c + cone_c + slack_c

        tangent_inequality_mat_pad = lambda x: tensor_utils.pad_mat_right(x, 0, 2 * k)
        slack_inequality_mat_pad = lambda x: tensor_utils.pad_mat_right(x, 0, k)


        eye_k = torch.eye(k).unsqueeze(0).repeat(batch_n, 1, 1)

        # Nonnegativity
        R_nonneg = -eye_k
        h_nonneg = torch.zeros(k, 1).unsqueeze(0).repeat(batch_n, 1, 1)

        R_nonneg = slack_inequality_mat_pad(tangent_inequality_mat_pad(R_nonneg))

        # Nonpenetration
        R_nonpen = torch.cat((-sp.dt * Jn.bmm(gamma).bmm(M_i).bmm((J_tilde.bmm(gamma)).transpose(1,2)), -eye_k), dim=2)
        h_nonpen = phi + sp.dt * Jn.bmm(gamma).bmm(f)

        # Coulomb friction constraint
        E_cf = tensor_utils.veceye(k, bases_n).unsqueeze(0).repeat(batch_n, 1, 1)
        R_coulomb = torch.cat((-E_cf, G.transpose(1,2)), dim=2)
        h_coulomb = torch.zeros(batch_n, bases_n * k, 1)

        R_coulomb = slack_inequality_mat_pad(R_coulomb)

        R = torch.cat((R_nonneg, R_nonpen, R_coulomb), dim=1)
        h = torch.cat((h_nonneg, h_nonpen, h_coulomb), dim=1)

        try:
            full_sol = solver_utils.solve_qp_osqpth(2 * A, b, R, h).unsqueeze(2)
        except:
            print('QP solve fail')
            return torch.tensor([[[0.0]]])

        # sum in case batch_n > 1
        qp_loss = utils.compute_quadratic_loss(A, b, c, full_sol).sum()
        qp_loss = qp_loss * (torch.norm(F_data, 2, dim=1).unsqueeze(2) > config.w_contact_threshold).int()
        qp_loss = qp_loss.sum()
        
        loss_terms = [utils.compute_quadratic_loss(comp_n_A, comp_n_b, comp_n_c, full_sol).sum(),
                      utils.compute_quadratic_loss(comp_t_A, comp_t_b, comp_t_c, full_sol).sum(),
                      utils.compute_quadratic_loss(match_A, match_b, match_c, full_sol).sum(),
                      utils.compute_quadratic_loss(cone_A, cone_b, cone_c, full_sol).sum(),
                      utils.compute_quadratic_loss(slack_A, slack_b, slack_c, full_sol).sum()]

        regularizers = []

        ##### penalize penetration:
        phi_penalizer = lambda phi: torch.sum(torch.clamp(-phi, min=0) ** 2)
        pen_loss = config.w_penetration * (phi_penalizer(phi) + phi_penalizer(phi_plus))
        regularizers.append(pen_loss)

        ##### constrain config grad normal:
        # L1 cost constraining phi norms w.r.t configuration to be one
        pos_norms = torch.norm(Jn[:, :, 0:3], dim=2)
        grad_normal_loss = config.w_config_grad_normal * \
                           ((pos_norms - torch.ones(pos_norms.shape)) ** 2).sum()
        regularizers.append(grad_normal_loss)

        ##### constrain config grad tangent:
        # L1 cost constraining phi_t norms w.r.t configuration to be one
        # NOTE: THIS IS BROKEN, DOESN'T NEED TO BE UNIT, NEEDS TO BE MU!!
        regularizers.append(torch.tensor(0.0))


        ##### constrain config grad perp:
        # L1 cost constraining phi_t norms perpendicular to phi norms
        if torch.norm(Jt_tilde) == 0.0:
            grad_perp_loss = torch.tensor(0.0)
        else:
            def normalize(vecs: Tensor) -> Tensor:
                norms = vecs.norm(dim=2).unsqueeze(2).repeat(1, 1, 3)
                return vecs / norms
            pos_normals = normalize(Jn[:, :, 0:3])
            pos_normals = pos_normals.repeat(1,1,2).reshape(batch_n, k * 2, 3)
            pos_tangents = normalize(Jt_tilde[:, :, 0:3])

            grad_perp_loss = config.w_config_grad_perp * \
                             ((pos_normals * pos_tangents).sum(dim=2) ** 2).sum()
        regularizers.append(grad_perp_loss)


        ##### constrain st estimate normal:
        # L2 cost on phi_plus_hat deviating from phi_plus
        phi_plus_hat = phi + sp.dt * Jn.bmm(gamma).bmm(velocity_plus)
        st_pen_loss = config.w_st_estimate_pen * \
                       torch.sum(torch.clamp(-phi_plus_hat, min=0) ** 2)#/batch_n
        regularizers.append(st_pen_loss)

        phi_norm = (torch.norm(phi_plus - phi_plus_hat, dim=1) ** 2).sum()#/batch_n
        st_normal_loss = config.w_st_estimate_normal * phi_norm
        regularizers.append(st_normal_loss)


        ##### constrain st estimate tangent:
        phi_t = self.interaction.compute_phi_t_previous()
        phi_t_plus = self.interaction.compute_phi_t_history()
        phi_t_plus_hat = phi_t + sp.dt * Jt_tilde.bmm(gamma).bmm(velocity_plus)
        phi_t_norm = (torch.norm(phi_t_plus - phi_t_plus_hat, dim=1) ** 2).sum()
        st_tangent_loss = config.w_st_estimate_tangent * phi_t_norm
        regularizers.append(st_tangent_loss)

        # Penalize second derivative of tangent jacobian
        Jt_tilde_plus = self.interaction.compute_Jt_tilde_history()
        delta_vc = Jt_tilde_plus.bmm(gamma).bmm(velocity_plus) - \
                   Jt_tilde.bmm(gamma).bmm(velocity_plus)
        vc_norm = (torch.norm(delta_vc, dim=1) ** 2).sum()#/batch_n
        tangent_jac_d2_loss = config.w_tangent_jac_d2 * vc_norm
        regularizers.append(tangent_jac_d2_loss)

        total_loss = qp_loss + 0 # Make new variable by adding 0
        for regularizer in regularizers:
            total_loss = total_loss + regularizer

        return [total_loss, qp_loss] + loss_terms + regularizers

    def compute_G_bases(self, bases_n):
        bases = torch.zeros(bases_n, 2)
        for i, angle in enumerate(np.linspace(0, 2 * math.pi * (1 - 1 / bases_n), bases_n)):
            bases[i, 0] = math.cos(angle)
            bases[i, 1] = math.sin(angle)

        return bases

    def compute_G(self) -> Tensor:
        bases = self.G_bases
        # self.interaction.k()
        G = tensor_utils.block_diag(bases.t().unsqueeze(0).repeat(self.interaction.k(), 1, 1))
        G = G.unsqueeze(0).repeat(self.interaction.batch_n(), 1, 1)
        return G
