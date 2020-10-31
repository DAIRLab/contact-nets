import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from typing import *

from abc import ABC, abstractmethod

from contactnets.entity import Entity

from contactnets.utils import utils, tensor_utils

import pdb

class Interaction(Module, ABC):
    entities: List[Entity]

    def __init__(self, entities: List[Entity]) -> None:
        super(Interaction, self).__init__()
        self.entities = entities

    # Entities hold a list of state histories.
    # compute_xxx_history computes a value for the last time step (by default, if i=-1)
    # compute_xxx_previous is a convenience function for computing the second-most-recent
    # time step

    @abstractmethod
    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        # Return dimension: batch_n x n_verts x 1
        pass

    def compute_phi_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_phi(configurations)

    def compute_phi_previous(self) -> Tensor:
        return self.compute_phi_history(i=-2)


    @abstractmethod
    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        # Return dimension: batch_n x n_verts x configuration_n
        pass

    def compute_Jn_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_Jn(configurations)

    def compute_Jn_previous(self) -> Tensor:
        return self.compute_Jn_history(i=-2)


    @abstractmethod
    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        # Return dimension: batch_n x n_verts x 1
        pass

    def compute_phi_t_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_phi_t(configurations)

    def compute_phi_t_previous(self) -> Tensor:
        return self.compute_phi_t_history(i=-2)


    @abstractmethod
    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        # Return dimension: batch_n x n_verts x configuration_n
        pass

    def compute_Jt_tilde_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_Jt_tilde(configurations)

    def compute_Jt_tilde_previous(self) -> Tensor:
        return self.compute_Jt_tilde_history(i=-2)


    @abstractmethod
    def k(self) -> int:
        # Number of elements in phi (aka number of vertices)
        pass


    def compute_M(self, configurations: List[Tensor]) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M(configuration) \
                            for entity, configuration in zip(self.entities, configurations)])

    def compute_M_history(self, i=-1) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M_history(i=i) \
                                        for entity in self.entities])

    def compute_M_previous(self) -> Tensor:
        return self.compute_M_history(i=-2)


    def compute_M_i(self, configurations: List[Tensor]) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M_i(configuration) \
                            for entity, configuration in zip(self.entities, configurations)])

    def compute_M_i_history(self, i=-1) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M_i_history(i=i) \
                                        for entity in self.entities])

    def compute_M_i_previous(self) -> Tensor:
        return self.compute_M_i_history(i=-2)


    def compute_gamma(self, configurations: List[Tensor]) -> Tensor:
        return tensor_utils.block_diag([entity.compute_gamma(configuration) \
                            for entity, configuration in zip(self.entities, configurations)])

    def compute_gamma_history(self, i=-1) -> Tensor:
        return tensor_utils.block_diag([entity.compute_gamma_history(i=i) \
                                        for entity in self.entities])

    def compute_gamma_previous(self) -> Tensor:
        return self.compute_gamma_history(i=-2)


    def compute_f(self, sp, configurations: List[Tensor], velocities: List[Tensor],
                  controls: List[Tensor], dt=None) -> Tensor:
        # TODO: return list?
        f_list = utils.filter_none([
                            entity.compute_f(sp, configuration, velocity, control, dt=dt) \
                            for entity, configuration, velocity, control in \
                            zip(self.entities, configurations, velocities, controls)])

        return torch.cat(f_list, dim=1)

    def compute_f_history(self, sp, i=-1, dt=None) -> Tensor:
        f_list = utils.filter_none([entity.compute_f_history(sp, i=i, dt=dt) \
                                    for entity in self.entities])

        return torch.cat(f_list, dim=1)

    def compute_f_previous(self, sp) -> Tensor:
        return self.compute_f_history(sp, i=-2)


    def batch_n(self) -> int:
        batch_ns = [entity.batch_n() for entity in self.entities]
        assert(utils.elements_identical(batch_ns))
        return batch_ns[0]

class DirectInteraction(Module, ABC):
    # Used for deep end-to-end / unstructured learning
    entities: List[Entity]

    def __init__(self, entities: List[Entity]) -> None:
        super(DirectInteraction, self).__init__()
        self.entities = entities

    @abstractmethod
    def compute_impulses(self, sp) -> List[Tensor]:
        pass
