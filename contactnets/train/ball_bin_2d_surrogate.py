from dataclasses import dataclass
import pdb  # noqa
from typing import List

import torch
from torch import Tensor

from contactnets.interaction import BallBin2D
from contactnets.system import System
from contactnets.train import Loss, LossType
from contactnets.utils import solver_utils, tensor_utils, utils


@dataclass
class BinSurrogateConfig2D:
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

    robust_sqrt:            bool = False


class BallBin2DSurrogate(Loss):
    interaction: BallBin2D
    G_bases: Tensor
    penalize_penetration: bool
    constrain_config_grad_normal: bool
    constrain_config_grad_tangent: bool
    constrain_st_estimate_normal: bool
    constrain_st_estimate_tangent: bool

    def __init__(self, system: System, interaction: BallBin2D,
                 G_bases: Tensor = None, config = BinSurrogateConfig2D()) -> None:
        super().__init__(
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
            G_bases = torch.tensor([[1.], [-1.]])
        self.G_bases = G_bases

        k = self.interaction.contact_n()
        self.G = tensor_utils.block_diag(G_bases.t().unsqueeze(0).repeat(k, 1, 1))
        self.G[torch.abs(self.G) < 0.5] = 0.

        bases_n = self.G_bases.shape[0]

        E_b = tensor_utils.veceye(k, bases_n)

        self.EG = torch.cat((E_b.t(), self.G))

        self.config = config

    def compute_loss(self, meta_save_file: str = None) -> List[Tensor]:
        sp = self.system.params
        config = self.config

        # Get configuration / velocity for polygon
        velocity = torch.cat(self.interaction.get_velocity_previous(), dim=1)
        velocity_plus = torch.cat(self.interaction.get_velocity(), dim=1)
        control = torch.cat(self.interaction.get_control_previous(), dim=1)

        batch_n = self.interaction.batch_n()
        ball_n = self.interaction.ball_n()
        bases_n = self.G_bases.shape[0]
        k = self.interaction.contact_n()

        G = self.compute_G()

        phi = self.interaction.compute_phi_previous()
        phi_plus = self.interaction.compute_phi_history()

        Jn = self.interaction.compute_Jn_previous()
        Jt_tilde = self.interaction.compute_Jt_tilde_previous()
        Jt = G.transpose(1, 2).bmm(Jt_tilde)
        J = torch.cat((Jn, Jt), dim=1)  # noqa
        J_tilde = torch.cat((Jn, Jt_tilde), dim=1)

        gamma = self.interaction.compute_gamma_previous()
        f = self.interaction.compute_f_previous(sp)

        M = self.interaction.compute_M_previous()
        M_i = self.interaction.compute_M_i_previous()

        gravity_dir = torch.tensor([0, -1, 0]).reshape(1, 3, 1).repeat(batch_n, ball_n, 1)
        gravity_dv = sp.g * sp.dt * gravity_dir

        v_diff = velocity_plus - velocity

        F_data = M.bmm(v_diff - gravity_dv - control)

        # Optimization variables are lambda_n, lambda_t

        def normal_mat_pad(x): return tensor_utils.diag_prepend(x, 0, k)
        def normal_vec_pad(x): return tensor_utils.pad_left(x, 0, k)

        def tangent_mat_pad(x): return tensor_utils.diag_append(x, 0, k)
        def tangent_vec_pad(x): return tensor_utils.pad_right(x, 0, k)

        def slack_mat_pad(x): return tensor_utils.diag_append(x, 0, k)
        def slack_vec_pad(x): return tensor_utils.pad_right(x, 0, k)

        # lambda_n and phi complementarity
        comp_n_A = config.w_comp_n * tensor_utils.matrix_diag((phi_plus.squeeze(2) ** 2))
        comp_n_A = tangent_mat_pad(comp_n_A)
        comp_n_A = slack_mat_pad(comp_n_A)

        # lambda_t and phi complementarity
        phi_expand = phi_plus
        comp_t_A = config.w_comp_t * tensor_utils.matrix_diag((phi_expand.squeeze(2) ** 2))
        comp_t_A = normal_mat_pad(comp_t_A)
        comp_t_A = slack_mat_pad(comp_t_A)

        # Match impulse data (multiply by M_i to get scaling)
        # Term of form (M_i gamma^T [Jn, Jt_tilde]^T lambda - M_i F)^2
        match_quad_A = M_i.bmm(gamma.transpose(1, 2)).bmm(J_tilde.transpose(1, 2))
        match_quad_b = M_i.bmm(F_data)
        match_A = config.w_match * match_quad_A.transpose(1, 2).bmm(match_quad_A)
        match_b = config.w_match * (-2) * match_quad_b.transpose(1, 2).bmm(match_quad_A)
        match_c = config.w_match * match_quad_b.transpose(1, 2).bmm(match_quad_b)

        match_A = slack_mat_pad(match_A)
        match_b = slack_vec_pad(match_b)

        # Friction cone boundary
        sliding_vels = Jt_tilde.bmm(gamma).bmm(velocity_plus)
        cone_normal_mat = tensor_utils.matrix_diag(sliding_vels.squeeze(2))

        sliding_vel_norms = sliding_vels.mul(sliding_vels)

        if config.robust_sqrt:
            sliding_vel_norms = tensor_utils.robust_sqrt(sliding_vel_norms)
        else:
            sliding_vel_norms = torch.sqrt(sliding_vel_norms)
        cone_tangent_mat = tensor_utils.matrix_diag(sliding_vel_norms.squeeze(2))
        cone_mat = torch.cat((cone_normal_mat, cone_tangent_mat), dim=2)

        cone_A = config.w_cone * cone_mat.transpose(1, 2).bmm(cone_mat)
        cone_A = slack_mat_pad(cone_A)

        # Penalize slack variables (penetration phi)
        slack_A = config.w_penetration_slack * tensor_utils.diag_append(
            torch.zeros(k * 2, k * 2).unsqueeze(0).repeat(batch_n, 1, 1), 1.0, k)
        slack_b = config.w_penetration_slack * torch.zeros(batch_n, 1, 3 * k)
        slack_c = config.w_penetration_slack * torch.zeros(batch_n, 1, 1)

        A = comp_n_A + comp_t_A + match_A + cone_A + slack_A
        b = match_b + slack_b
        c = match_c + slack_c

        def tangent_inequality_mat_pad(x): return tensor_utils.pad_right(x, 0, k)
        def slack_inequality_mat_pad(x): return tensor_utils.pad_right(x, 0, k)

        eye_k = torch.eye(k).unsqueeze(0).repeat(batch_n, 1, 1)

        # Nonnegativity
        R_nonneg = -eye_k
        h_nonneg = torch.zeros(k, 1).unsqueeze(0).repeat(batch_n, 1, 1)

        R_nonneg = slack_inequality_mat_pad(tangent_inequality_mat_pad(R_nonneg))

        # Nonpenetration
        Jt_gamma_transpose = (J_tilde.bmm(gamma)).transpose(1, 2)
        nonpen_intermediate = Jn.bmm(gamma).bmm(M_i).bmm(Jt_gamma_transpose)
        R_nonpen = torch.cat((-sp.dt * nonpen_intermediate, -eye_k), dim=2)
        h_nonpen = phi + sp.dt * Jn.bmm(gamma).bmm(f)

        # Coulomb friction constraint
        E_cf = tensor_utils.veceye(k, bases_n).unsqueeze(0).repeat(batch_n, 1, 1)
        R_coulomb = torch.cat((-E_cf, G.transpose(1, 2)), dim=2)
        h_coulomb = torch.zeros(batch_n, bases_n * k, 1)

        R_coulomb = slack_inequality_mat_pad(R_coulomb)

        R = torch.cat((R_nonneg, R_nonpen, R_coulomb), dim=1)
        h = torch.cat((h_nonneg, h_nonpen, h_coulomb), dim=1)

        try:
            full_sol = solver_utils.solve_qp_osqpth(2 * A, b, R, h)
        except Exception:
            print('QP solve fail')
            return [torch.tensor([[[0.0]]])]

        # sum in case batch_n > 1
        qp_loss = utils.compute_quadratic_loss(A, b, c, full_sol).sum()
        contact_mask = torch.norm(F_data, 2, dim=1).unsqueeze(2) > config.w_contact_threshold
        qp_loss = qp_loss * contact_mask.int()
        qp_loss = qp_loss.sum()

        nz = 3 * k
        b_zero = config.w_comp_t * torch.zeros(batch_n, 1, nz)
        c_zero = config.w_comp_t * torch.zeros(batch_n, 1, 1)
        loss_terms = [utils.compute_quadratic_loss(comp_n_A, b_zero, c_zero, full_sol).sum(),
                      utils.compute_quadratic_loss(comp_t_A, b_zero, c_zero, full_sol).sum(),
                      utils.compute_quadratic_loss(match_A, match_b, match_c, full_sol).sum(),
                      utils.compute_quadratic_loss(cone_A, b_zero, c_zero, full_sol).sum()]
        loss_terms += [utils.compute_quadratic_loss(slack_A, slack_b, slack_c,
                                                    full_sol).sum()]

        regularizers = []

        ##### penalize penetration:
        def phi_penalizer(phi): return torch.sum(torch.clamp(-phi, min=0) ** 2)
        # pen_loss = config.w_penetration * (phi_penalizer(phi) + phi_penalizer(phi_plus))
        pen_loss = config.w_penetration * phi_penalizer(phi_plus)
        regularizers.append(pen_loss)

        ##### constrain config grad normal:
        # L1 cost constraining phi norms w.r.t configuration to be one
        # pos_norms = torch.norm(Jn[:, :, 0:3], dim=2)
        # grad_normal_loss = config.w_config_grad_normal * \
        #                   ((pos_norms - torch.ones(pos_norms.shape)) ** 2).sum()

        # TODO: reimplement for multiobject
        grad_normal_loss = config.w_config_grad_normal * torch.tensor(0.0)
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
            # pos_normals = normalize(Jn[:, :, 0:3])
            # pos_tangents = normalize(Jt_tilde[:, :, 0:3])

            # grad_perp_loss = config.w_config_grad_perp * \
            #                ((pos_normals * pos_tangents).sum(dim=2) ** 2).sum()

            # TODO: reimplement for multiobject
            grad_perp_loss = torch.tensor(0.0)
        regularizers.append(grad_perp_loss)

        ##### constrain st estimate normal:
        # L2 cost on phi_plus_hat deviating from phi_plus
        phi_plus_hat = phi + sp.dt * Jn.bmm(gamma).bmm(velocity_plus)
        st_pen_loss = config.w_st_estimate_pen * \
            torch.sum(torch.clamp(-phi_plus_hat, min=0) ** 2)  # /batch_n
        regularizers.append(st_pen_loss)

        phi_norm = (torch.norm(phi_plus - phi_plus_hat, dim=1) ** 2).sum()  # /batch_n
        st_normal_loss = config.w_st_estimate_normal * phi_norm
        regularizers.append(st_normal_loss)


        ##### constrain st estimate tangent:
        # phi_t = self.interaction.compute_phi_t_previous()
        # phi_t_plus = self.interaction.compute_phi_t_history()
        # phi_t_plus_hat = phi_t + sp.dt * Jt_tilde.bmm(gamma).bmm(velocity_plus)
        # phi_t_norm = (torch.norm(phi_t_plus - phi_t_plus_hat, dim=1) ** 2).sum()
        st_tangent_loss = config.w_st_estimate_tangent * torch.tensor(0.0)
        regularizers.append(st_tangent_loss)

        # Penalize second derivative of tangent jacobian
        Jt_tilde_plus = self.interaction.compute_Jt_tilde_history()
        delta_vc = (Jt_tilde_plus - Jt_tilde).bmm(gamma).bmm(velocity_plus)
        vc_norm = (torch.norm(delta_vc, dim=1) ** 2).sum()  # /batch_n
        tangent_jac_d2_loss = config.w_tangent_jac_d2 * vc_norm
        regularizers.append(tangent_jac_d2_loss)

        total_loss = qp_loss + 0  # Make new variable by adding 0
        for regularizer in regularizers:
            total_loss = total_loss + regularizer

        return [total_loss, qp_loss] + loss_terms + regularizers

    def compute_G(self) -> Tensor:
        return self.G.unsqueeze(0).repeat(self.interaction.batch_n(), 1, 1)

    def compute_EG(self) -> Tensor:
        return self.EG.unsqueeze(0).repeat(self.interaction.batch_n(), 1, 1)
