import pdb  # noqa
from typing import TYPE_CHECKING, List

import torch
from torch import Tensor
from torch.nn import Module

from contactnets.entity import Dynamic3D, Ground3D
from contactnets.interaction import InteractionResolver
from contactnets.utils import solver_utils, tensor_utils

if TYPE_CHECKING:
    from contactnets.system import System


class ElasticLCP(InteractionResolver):
    """A modified linear complimentarity program resolver for a single elastic interaction."""
    G_bases: Tensor
    restitution: Tensor
    restitute_friction: bool
    dt_eps: float

    def __init__(self, interactions: List[Module], G_bases: Tensor,
                 restitution: Tensor, restitute_friction: bool) -> None:
        super().__init__(interactions, [])
        assert(len(interactions) == 1)

        self.G_bases = G_bases
        self.restitution = restitution
        self.restitute_friction = restitute_friction
        self.dt_eps = 1e-5

    def step(self, system: 'System') -> None:
        sp = system.params
        interaction = self.interactions[0]

        batch_n = interaction.batch_n()
        gamma = interaction.compute_gamma_history()
        Jn = interaction.compute_Jn_history().bmm(gamma)
        Jt_tilde = interaction.compute_Jt_tilde_history().bmm(gamma)
        J = torch.cat((Jn, Jt_tilde), dim=1)  # noqa

        (q, v, impulses) = self.compute_corner_impulses(sp)

        for entity in interaction.entities:
            if isinstance(entity, Dynamic3D):
                entity.append_state(q, v)
            elif isinstance(entity, Ground3D):
                entity.append_state(torch.zeros(batch_n, 0, 1), torch.zeros(batch_n, 0, 1))
            else:
                raise RuntimeError('Not supported entity')

    def velocity_update_continuous(self, sp, dts, velocities):
        interaction = self.interactions[0]
        configurations = [entity.get_configuration() for entity in interaction.entities]
        controls = [entity.get_control() for entity in interaction.entities]
        dt = dts.reshape(1, 1, 1).repeat(interaction.batch_n(), 1, 1)
        return interaction.compute_f(sp, configurations, [velocities, None], controls, dt=dt)

    def velocity_update_null(self, sp, dts, velocities):
        return velocities

    def restitution_net_impulse(self, sp, configurations, velocities, restitution_impulses):
        interaction = self.interactions[0]
        configurations = [configurations, None]
        gamma = interaction.compute_gamma(configurations=configurations)
        Jn = interaction.compute_Jn(configurations=configurations).bmm(gamma)
        Jt_tilde = interaction.compute_Jt_tilde(configurations=configurations).bmm(gamma)
        J = torch.cat((Jn, Jt_tilde), dim=1)
        restitution_impulse = J.transpose(1, 2).bmm(restitution_impulses)
        return restitution_impulse

    def restitution_impulses(self, sp, compression_impulses):
        interaction = self.interactions[0]
        impulses = compression_impulses.clone()
        if not self.restitute_friction:
            impulses[:, interaction.contact_n():, :] *= 0.
        return impulses * self.restitution


    def calculate_impact_times(self, phis, new_phis, delta_active_contact, dts):
        impact_dts = dts.clone()
        impactors = delta_active_contact.sum(dim=1)
        impacting_contacts = -torch.ones(phis.shape[0]).long()
        for i in range(phis.shape[0]):
            if impactors[i]:
                new_i = delta_active_contact[i, :]
                candidate_impacting = torch.arange(0, phis.shape[1])[new_i]
                hit_times = phis[i, new_i] / (phis[i, new_i] - new_phis[i, new_i])
                hit_times = hit_times.clamp(min=0., max=1.)

                impacting_contacts[i] = candidate_impacting[torch.argmin(hit_times)]
                impact_dts[i] = dts[i] * torch.min(hit_times)
        return (impact_dts, impacting_contacts)


    def compute_corner_impulses(self, sp, i=-1):
        interaction = self.interactions[0]

        batch_n = interaction.batch_n()
        k = interaction.contact_n()

        dts_left = torch.ones(batch_n) * sp.dt


        configurations = interaction.poly.get_configuration(i=i)
        velocities = interaction.poly.get_velocity(i=i)
        phis = interaction.compute_phi(configurations=[configurations, None]).squeeze(2)
        active_contacts = phis <= 0.
        active_sims = dts_left > 1e-10
        total_impulse = torch.zeros(batch_n, 3 * k, 1)
        while torch.any(active_sims) > 0:
            (new_configs, new_velocities, new_impulse, new_phis) = \
                self.compute_impact_free_step(sp, configurations, velocities, dts_left,
                                              active_contacts, active_sims,
                                              self.velocity_update_continuous)
            new_active_contacts = new_phis <= 0.
            delta_active_contact = (new_active_contacts) * (~active_contacts)

            impactors = delta_active_contact.sum(dim=1)
            (preimpact_dts, impacting_contacts) = self.calculate_impact_times(
                phis, new_phis, delta_active_contact, dts_left)

            # don't underestimate phi
            for i in range(batch_n):
                if impactors[i] > 0:
                    if preimpact_dts[i] < self.dt_eps:
                        preimpact_dts[i] = self.dt_eps
                    # active_contacts[i,impacting_contacts[i]] = True


            (preimpact_configs, preimpact_velocities, preimpact_impulse, preimpact_phis) = \
                self.compute_impact_free_step(sp, configurations, velocities, preimpact_dts,
                                              active_contacts, active_sims,
                                              self.velocity_update_continuous)

            active_contacts = preimpact_phis <= 0.
            # get updated contact set
            for i in range(batch_n):
                if impactors[i] > 0:
                    pass
                    # active_contacts[i,impacting_contacts[i]] = True

            (compression_configs, compression_velocities,
                compression_impulse, compression_phis) = \
                self.compute_impact_free_step(sp, preimpact_configs, preimpact_velocities,
                                              0.0 * preimpact_dts, active_contacts, active_sims,
                                              self.velocity_update_null)

            compressed_velocities = preimpact_velocities.clone()
            compressed_impulse = preimpact_impulse.clone()
            for i in range(batch_n):
                if impactors[i] > 0:
                    compressed_velocities[i, :, :] = compression_velocities[i, :, :]
                    compressed_impulse[i, :, :] += compression_impulse[i, :, :]

            restution_nominal_impulses = self.restitution_impulses(sp, compression_impulse)
            restution_nominal_net_impulse = self.restitution_net_impulse(
                sp, compression_configs, compression_velocities, restution_nominal_impulses)

            def restution_velocity(s, dt, v):
                return v + interaction.compute_M_i_history().bmm(restution_nominal_net_impulse)

            (rest_configs, rest_velocities, rest_impulse_correction, rest_phis) = \
                self.compute_impact_free_step(sp, compression_configs, compression_velocities,
                                              0.0 * preimpact_dts, active_contacts,
                                              active_sims, restution_velocity)

            new_velocities = compression_velocities.clone()
            new_impulse = compressed_impulse.clone()
            for i in range(batch_n):
                if impactors[i] > 0:
                    new_velocities[i, :, :] = rest_velocities[i, :, :]
                    new_impulse[i, :, :] += restution_nominal_impulses[i, :, :]
                    new_impulse[i, :, :] += rest_impulse_correction[i, :, :]

            dts_left -= preimpact_dts
            total_impulse += new_impulse

            # add cloned contact
            active_sims = dts_left > 1e-10
            phis = new_phis.clone()
            configurations = preimpact_configs.clone()
            velocities = new_velocities.clone()

        return (new_configs, new_velocities, total_impulse)

    def compute_impact_free_step(self, sp, configurations, velocities, dts,
                                 active_contacts, active_sims, velocity_update):

        interaction = self.interactions[0]
        k = interaction.contact_n()
        bases_n = self.G_bases.shape[0]
        batch_n = interaction.batch_n()

        N_active_contact = active_contacts.sum(dim=1)

        new_impulse_n = torch.zeros(batch_n, k)
        new_impulse_t = torch.zeros(batch_n, bases_n * k)

        Id = torch.eye(k).unsqueeze(0).repeat(1, 1, 1)

        f = velocity_update(sp, dts, velocities)

        for i in range(batch_n):
            if active_sims[i] and N_active_contact[i] > 0:
                (Mi, wi) = self.compute_lcp_step(sp, active_contacts[i, :],
                                                 configurations[i:(i + 1), :, :],
                                                 velocities[i:(i + 1), :, :], f, dts[i])
                ki = N_active_contact[i]
                sols = self.compute_impulse_lcp(Mi, wi)
                lambda_n = sols[:, 0:ki, :]
                lambda_t = sols[:, ki:ki + ki * bases_n, :]
                inflate = Id[:, :, active_contacts[i, :]]
                lambda_n = inflate.bmm(lambda_n)
                # TODO: fix inflation
                # lambda_t = E.bmm(inflate.transpose(1,2)).bmm(lambda_t)
                lti = active_contacts[i, :].unsqueeze(1).repeat(1, bases_n)\
                    .reshape(bases_n * interaction.contact_n(), 1).squeeze()
                # if active_contacts.sum() > 1:
                #     pdb.set_trace()
                new_impulse_n[i, :] = lambda_n.view(-1)
                new_impulse_t[i, lti] = lambda_t.view(-1)

        new_impulse = torch.cat((new_impulse_n.unsqueeze(2),
                                 self.compute_G().bmm(new_impulse_t.unsqueeze(2))), dim=1)

        configs = [configurations, None]

        gamma = interaction.compute_gamma(configurations=configs)
        Jn_tilde = interaction.compute_Jn(configurations=configs)
        Jn = Jn_tilde.bmm(gamma)
        Jt_tilde = interaction.compute_Jt_tilde(configurations=configs)
        Jt = Jt_tilde.bmm(gamma)
        J = torch.cat((Jn, Jt), dim=1)

        net_impulse = J.transpose(1, 2).bmm(new_impulse)
        velocity_n = f + interaction.compute_M_i(configurations=configs).bmm(net_impulse)
        configuration_n = configurations + interaction.compute_gamma(
            configurations=configs).bmm(velocity_n)\
            .bmm(dts.reshape(batch_n, 1, 1))
        quat_norm = configuration_n[:, 3:7, :].norm(dim = 1)
        configuration_n[:, 3:7, :] /= quat_norm

        new_phis = interaction.compute_phi(configurations=[configuration_n, None]).squeeze(2)

        return (configuration_n, velocity_n, new_impulse, new_phis)

    def compute_impulse_lcp(self, lcp_mat, lcp_vec):
        lcp_mat = lcp_mat
        lcp_vec = lcp_vec
        Id = torch.eye(lcp_mat.shape[1])
        Id = Id.unsqueeze(0).repeat(1, 1, 1)
        lcp_mat = lcp_mat + Id * 0.001

        for mat in lcp_mat:
            real_eigs = torch.eig(mat).eigenvalues[:, 0]
            assert(torch.all(real_eigs > -torch.ones(real_eigs.shape) * 0.01))

        return solver_utils.solve_lcp_batch(lcp_mat, lcp_vec)

    def compute_lcp_step(self, sp, active_contacts, configurations, velocities, f, dts):
        interaction = self.interactions[0]
        bases_n = self.G_bases.shape[0]
        batch_n = interaction.batch_n()
        k = active_contacts.sum()

        configs = [configurations, None]

        gamma = interaction.compute_gamma(configurations=configs)
        Jn = interaction.compute_Jn(configurations=configs)[:, active_contacts, :].bmm(gamma)
        doubled_ac = active_contacts.unsqueeze(1).repeat(1, bases_n)\
            .reshape(bases_n * interaction.contact_n(), 1).squeeze()
        Jt = self.compute_Jt(configurations=configurations)[:, doubled_ac, :].bmm(gamma)
        interaction = self.interactions[0]


        LCP_vec = torch.cat((Jn.bmm(f), Jt.bmm(f), torch.zeros((batch_n, k, 1))), dim=1)

        return self.compute_lcp_matrix(sp, active_contacts, configurations), LCP_vec

    def compute_lcp_matrix(self, sp, active_contacts, configurations):
        interaction = self.interactions[0]
        bases_n = self.G_bases.shape[0]
        batch_n = interaction.batch_n()
        k = active_contacts.sum()

        E = tensor_utils.veceye(k, bases_n).unsqueeze(0).repeat(batch_n, 1, 1)

        configs = [configurations, None]
        gamma = interaction.compute_gamma(configurations=configs)
        M_i = interaction.compute_M_i_history()
        Jn_tilde = interaction.compute_Jn(configurations=configs)[:, active_contacts, :]
        Jn = Jn_tilde.bmm(gamma)
        doubled_ac = active_contacts.unsqueeze(1).repeat(1, bases_n)\
            .reshape(bases_n * interaction.contact_n(), 1).squeeze()
        Jt_tilde = self.compute_Jt(configurations=configurations)[:, doubled_ac, :]
        Jt = Jt_tilde.bmm(gamma)


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

        return LCP_mat

    def compute_Jt(self, i=-1, configurations=None):
        interaction = self.interactions[0]

        Jt_tilde = interaction.compute_Jt_tilde(configurations=[configurations, None])

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
