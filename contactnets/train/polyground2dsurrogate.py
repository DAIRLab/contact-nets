import torch
from torch import Tensor
from torch.nn import Module

from abc import ABC, abstractmethod
from typing import *

from contactnets.interaction import Interaction, PolyGround2D, LCP
from contactnets.system import System
from contactnets.train import Loss, LossType

from contactnets.utils import utils, solver_utils, tensor_utils

from dataclasses import dataclass

import pdb

@dataclass
class SurrogateConfig2D:
    w_penetration:          float = 0.0
    w_config_grad_normal:   float = 0.0
    w_config_grad_tangent:  float = 0.0
    w_config_grad_perp:     float = 0.0
    w_st_estimate_pen:      float = 0.0
    w_st_estimate_normal:   float = 0.0
    w_st_estimate_tangent:  float = 0.0
    w_tangent_jac_d2:       float = 0.0

    phi_penalties_lin:      bool  = False

class PolyGround2DSurrogate(Loss):
    interaction: PolyGround2D
    G_bases: Tensor
    config: SurrogateConfig2D

    def __init__(self, system: System,
                       interaction: PolyGround2D,
                       G_bases: Tensor = None,
                       config = SurrogateConfig2D()) -> None:
        super(PolyGround2DSurrogate, self).__init__(
                ['surrogate', 'surr_qp', 'surr_pen', 'surr_grad_normal',
                 'surr_grad_tangent', 'surr_grad_perp', 'surr_st_pen',
                 'surr_st_normal', 'surr_st_tangent', 'surr_tangent_d2'],
                system, LossType.STEPWISE, False)

        self.interaction = interaction

        if G_bases is None:
            # Can't be in argument b/c messes default tensor type
            G_bases = Tensor([1, -1])
        self.G_bases = G_bases

        self.config = config

    def compute_loss(self, meta_save_file: str = None) -> List[Tensor]:
        sp = self.system.params
        entity = self.interaction.poly

        # Get configuration / velocity for polygon / point
        configuration = entity.get_configuration_previous()
        configuration_plus = entity.get_configuration()
        velocity = entity.get_velocity_previous()
        velocity_plus = entity.get_velocity()
        control = entity.get_control_previous()

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
        f = entity.compute_f_previous(sp)

        M = self.interaction.compute_M_previous()
        M_i = self.interaction.compute_M_i_previous()

        control_dv = M_i.bmm(control) * sp.dt
        # Depending on point or poly gravity vec might need two or three elements
        gravity_dir = torch.tensor([0.0, -1, 0])[0 : configuration.shape[1]]
        gravity_dv = sp.g * gravity_dir.reshape(1, configuration.shape[1], 1) \
                                       .repeat(batch_n, 1, 1) * sp.dt

        v_diff = velocity_plus - velocity

        F_data = M.bmm(v_diff - gravity_dv - control)

        # Optimization variables are lambda_n, lambda_t
        normal_mat_pad = lambda x: tensor_utils.diag_prepend_mat(x, 0, k)
        normal_vec_pad = lambda x: tensor_utils.pad_mat_left(x, 0, k)

        tangent_mat_pad = lambda x: tensor_utils.diag_expand_mat(x, 0, k)
        tangent_vec_pad = lambda x: tensor_utils.pad_mat_right(x, 0, k)

        slack_mat_pad = lambda x: tensor_utils.diag_expand_mat(x, 0, k)
        slack_vec_pad = lambda x: tensor_utils.pad_mat_right(x, 0, k)

        # lambda_n and phi complementarity
        if self.config.phi_penalties_lin:
            comp_n_A = tensor_utils.matrix_diag(torch.clamp(phi_plus.squeeze(2), min=0.0))
        else:
            comp_n_A = tensor_utils.matrix_diag((phi_plus.squeeze(2) ** 2))
        comp_n_b = torch.zeros(batch_n, 1, k)
        comp_n_c = torch.zeros(batch_n, 1, 1)

        comp_n_A = slack_mat_pad(tangent_mat_pad(comp_n_A))
        comp_n_b = slack_vec_pad(tangent_vec_pad(comp_n_b))

        # lambda_t and phi complementarity
        if self.config.phi_penalties_lin:
            comp_t_A = tensor_utils.matrix_diag(torch.clamp(phi_plus.squeeze(2), min=0.0))
        else:
            comp_t_A = tensor_utils.matrix_diag((phi_plus.squeeze(2) ** 2))
        comp_t_b = torch.zeros(batch_n, 1, k)
        comp_t_c = torch.zeros(batch_n, 1, 1)

        comp_t_A = slack_mat_pad(normal_mat_pad(comp_t_A))
        comp_t_b = slack_vec_pad(normal_vec_pad(comp_t_b))

        # Match impulse data (multiply by M_i to get scaling)
        # Term of form (M_i gamma^T [Jn, Jt_tilde]^T lambda - M_i F)^2
        match_quad_A = M_i.bmm(gamma.transpose(1,2)).bmm(J_tilde.transpose(1,2))
        match_quad_b = M_i.bmm(F_data)
        match_A = match_quad_A.transpose(1, 2).bmm(match_quad_A)
        match_b = (-2) * match_quad_b.transpose(1, 2).bmm(match_quad_A)
        match_c = match_quad_b.transpose(1, 2).bmm(match_quad_b)

        match_A = slack_mat_pad(match_A)
        match_b = slack_vec_pad(match_b)

        # Friction cone boundary
        sliding_vels = Jt_tilde.bmm(gamma).bmm(velocity_plus)
        cone_normal_mat = tensor_utils.matrix_diag(sliding_vels.squeeze(2))
        cone_tangent_mat = tensor_utils.matrix_diag(torch.abs(sliding_vels.squeeze(2)))
        cone_mat = torch.cat((cone_normal_mat, cone_tangent_mat), dim=2)

        cone_A = cone_mat.transpose(1,2).bmm(cone_mat)
        cone_b = torch.zeros(batch_n, 1, 2 * k)
        cone_c = torch.zeros(batch_n, 1, 1)

        cone_A = slack_mat_pad(cone_A)
        cone_b = slack_vec_pad(cone_b)

        # Penalize slack variables (penetration phi)
        #if self.config.phi_penalties_lin:
        if False:
            slack_A = torch.zeros(3*k, 3*k).unsqueeze(0).repeat(batch_n, 1, 1)
            slack_b = torch.ones(batch_n, 1, 3*k)
            slack_c = torch.zeros(batch_n, 1, 1)
        else:
            slack_zeros = torch.zeros(2*k, 2*k).unsqueeze(0).repeat(batch_n, 1, 1)
            slack_A = tensor_utils.diag_expand_mat(slack_zeros, 1.0, k)
            slack_b = torch.zeros(batch_n, 1, 3*k)
            slack_c = torch.zeros(batch_n, 1, 1)

        A = comp_n_A + comp_t_A + match_A + cone_A + slack_A
        b = comp_n_b + comp_t_b + match_b + cone_b + slack_b
        c = comp_n_c + comp_t_c + match_c + cone_c + slack_c

        tangent_inequality_mat_pad = lambda x: tensor_utils.pad_mat_right(x, 0, k)
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

        full_sol = solver_utils.solve_qp_osqpth(2 * A, b, R, h).unsqueeze(2)
        # sum in case batch_n > 1
        qp_loss = utils.compute_quadratic_loss(A, b, c, full_sol).sum()


        regularizers = []

        ##### penalize penetration:
        if self.config.phi_penalties_lin:
            phi_penalizer = lambda phi: torch.sum(torch.clamp(-phi, min=0))
        else:
            phi_penalizer = lambda phi: torch.sum(torch.clamp(-phi, min=0) ** 2)
        pen_loss = self.config.w_penetration * (phi_penalizer(phi) + phi_penalizer(phi_plus))
        regularizers.append(pen_loss)


        ##### constrain config grad normal:
        # L1 cost constraining phi norms w.r.t configuration to be one
        pos_norms = torch.norm(Jn[:, :, 0:2], dim=2)
        grad_normal_loss = self.config.w_config_grad_normal * \
                           torch.abs(pos_norms - torch.ones(pos_norms.shape)).sum()
        regularizers.append(grad_normal_loss)


        ##### constrain config grad tangent:
        # L1 cost constraining phi_t norms w.r.t configuration to be one
        # NOTE: THIS IS BROKEN, DOESN'T NEED TO BE UNIT, NEEDS TO BE MU!!
        regularizers.append(torch.tensor(0.0))


        ##### constrain config grad perp:
        # L1 cost constraining phi_t norms perpendicular to phi norms
        def normalize(vecs: Tensor) -> Tensor:
            norms = vecs.norm(dim=2).unsqueeze(2).repeat(1, 1, 2)
            return vecs / norms

        pos_normals = normalize(Jn[:, :, 0:2])
        pos_tangents = normalize(Jt_tilde[:, :, 0:2])
        grad_perp_loss = self.config.w_config_grad_perp * \
                         torch.norm(pos_normals.bmm(pos_tangents.transpose(1,2)), 2)
        regularizers.append(grad_perp_loss)


        # constrain st estimate normal:
        # L2 cost on phi_plus_hat deviating from phi_plus
        phi_plus_hat = phi + sp.dt * Jn.bmm(gamma).bmm(velocity_plus)
        st_pen_loss = self.config.w_st_estimate_pen * \
                      torch.sum(torch.clamp(-phi_plus_hat, min=0) ** 2)
        regularizers.append(st_pen_loss)

        phi_norm = (torch.norm(phi_plus - phi_plus_hat, dim=1) ** 2).sum()
        st_normal_loss = self.config.w_st_estimate_normal * phi_norm
        regularizers.append(st_normal_loss)

        # Jn_plus = self.interaction.compute_Jn_history()
        # delta_vc = Jn_plus.bmm(gamma).bmm(velocity_plus) - Jn.bmm(gamma).bmm(velocity_plus)
        # vc_norm = (torch.norm(delta_vc, dim=1) ** 2).sum()


        # constrain st estimate tangent:
        phi_t = self.interaction.compute_phi_t_previous()
        phi_t_plus = self.interaction.compute_phi_t_history()
        phi_t_plus_hat = phi_t + sp.dt * Jt_tilde.bmm(gamma).bmm(velocity_plus)
        st_tangent_loss = self.config.w_st_estimate_tangent * \
                          (torch.norm(phi_t_plus - phi_t_plus_hat, dim=1) ** 2).sum()
        regularizers.append(st_tangent_loss)

        # Penalize second derivative of tangent jacobian
        Jt_tilde_plus = self.interaction.compute_Jt_tilde_history()
        delta_vc = Jt_tilde_plus.bmm(gamma).bmm(velocity_plus) - \
                   Jt_tilde.bmm(gamma).bmm(velocity_plus)
        vc_norm = (torch.norm(delta_vc, dim=1) ** 2).sum()
        tangent_jac_d2_loss = self.config.w_tangent_jac_d2 * vc_norm
        regularizers.append(tangent_jac_d2_loss)

        total_loss = qp_loss + 0 # Make new variable by adding 0
        for regularizer in regularizers:
            total_loss += regularizer

        return [total_loss, qp_loss] + regularizers

    def compute_G(self) -> Tensor:
        bases = self.G_bases
        # self.interaction.k()
        G = tensor_utils.block_diag(bases.t().unsqueeze(0).repeat(self.interaction.k(), 1, 1))
        G = G.unsqueeze(0).repeat(self.interaction.batch_n(), 1, 1)
        return G
