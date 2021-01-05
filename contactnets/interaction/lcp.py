import pdb  # noqa
from typing import TYPE_CHECKING, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from contactnets.interaction import InteractionResolver
from contactnets.utils import solver_utils, tensor_utils

if TYPE_CHECKING:
    from contactnets.system import System


class LCP(InteractionResolver):
    """A linear complimentarity program resolver for single interaction."""
    G_bases: Tensor

    def __init__(self, interactions: List[Module], G_bases: Tensor) -> None:
        super().__init__(interactions, [])
        assert(len(interactions) == 1)

        self.G_bases = G_bases

    def step(self, system: 'System') -> None:
        self.step_impulses(system, self.compute_impulses(system))

    def compute_impulses(self, system: 'System') -> List[Tensor]:
        sp = system.params
        interaction = self.interactions[0]

        gamma = interaction.compute_gamma_history()
        Jn = interaction.compute_Jn_history().bmm(gamma)
        Jt_tilde = interaction.compute_Jt_tilde_history().bmm(gamma)
        J = torch.cat((Jn, Jt_tilde), dim=1)

        net_impulses = J.transpose(1, 2).bmm(self.compute_corner_impulses(sp))

        impulses = []

        index = 0
        for entity in interaction.entities:
            impulses.append(net_impulses[:, index:index + entity.velocity_size(), :])
            index += entity.velocity_size()

        return impulses

    def compute_corner_impulses(self, sp, i=-1) -> Tensor:
        interaction = self.interactions[0]
        bases_n = self.G_bases.shape[0]

        batch_n = interaction.batch_n()
        k = interaction.contact_n()

        lcp_mat, lcp_vec = self.compute_lcp(sp, i=i)

        Id = torch.eye(lcp_mat.shape[1])
        Id = Id.unsqueeze(0).repeat(batch_n, 1, 1)
        lcp_mat = lcp_mat + Id * 0.001

        for mat in lcp_mat:
            real_eigs = torch.eig(mat).eigenvalues[:, 0]
            assert(torch.all(real_eigs > -torch.ones(real_eigs.shape) * 0.01))

        sols = solver_utils.solve_lcp_batch(lcp_mat, lcp_vec)

        lambda_n = sols[:, 0:k]
        lambda_t = sols[:, k:k + k * bases_n]

        lambda_t = self.compute_G().bmm(lambda_t)

        return torch.cat((lambda_n, lambda_t), dim=1)

    def compute_lcp(self, sp, i=-1) -> Tuple[Tensor, Tensor]:
        interaction = self.interactions[0]
        bases_n = self.G_bases.shape[0]

        batch_n = interaction.batch_n()
        k = interaction.contact_n()

        gamma = interaction.compute_gamma_history(i=i)
        M_i = interaction.compute_M_i_history(i=i)
        Jn = interaction.compute_Jn_history(i=i).bmm(gamma)
        Jt = self.compute_Jt(i=i).bmm(gamma)
        E = tensor_utils.veceye(k, bases_n).unsqueeze(0).repeat(batch_n, 1, 1)

        f = interaction.compute_f_history(sp, i=i)

        phi = interaction.compute_phi_history(i=i)

        # Compute block matrix terms
        tl = Jn.bmm(M_i).bmm(Jn.transpose(1, 2))
        tm = Jn.bmm(M_i).bmm(Jt.transpose(1, 2))
        tr = torch.zeros((batch_n, tl.shape[1], k))
        r1 = torch.cat((tl, tm, tr), dim=2)

        ml = Jt.bmm(M_i).bmm(Jn.transpose(1, 2))
        mm = Jt.bmm(M_i).bmm(Jt.transpose(1, 2))
        mr = E
        r2 = torch.cat((ml, mm, mr), dim=2)

        bl = torch.eye(k).unsqueeze(0).repeat(batch_n, 1, 1)
        bm = -E.transpose(1, 2)
        br = torch.zeros((batch_n, k, k))
        r3 = torch.cat((bl, bm, br), dim=2)

        LCP_mat = torch.cat((r1, r2, r3), dim=1)

        LCP_vec = torch.cat((Jn.bmm(f) + phi / sp.dt, Jt.bmm(f),
                             torch.zeros((batch_n, k, 1))), dim=1)

        return LCP_mat, LCP_vec

    def compute_Jt(self, i=-1) -> Tensor:
        interaction = self.interactions[0]

        Jt_tilde = interaction.compute_Jt_tilde_history(i=i)

        GT = self.compute_G()

        Jt = GT.transpose(1, 2).bmm(Jt_tilde)

        return Jt

    def compute_G(self) -> Tensor:
        interaction = self.interactions[0]
        contact_n = interaction.contact_n()
        bases = self.G_bases
        G = tensor_utils.block_diag(bases.t().unsqueeze(0).repeat(contact_n, 1, 1))
        G = G.unsqueeze(0).repeat(interaction.batch_n(), 1, 1)
        return G
