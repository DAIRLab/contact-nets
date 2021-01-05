from abc import ABC, abstractmethod
import pdb  # noqa
from typing import TYPE_CHECKING, List

import torch
from torch import Tensor
from torch.nn import Module

from contactnets.entity import Entity
from contactnets.utils import tensor_utils, utils

if TYPE_CHECKING:
    from contactnets.system import SystemParams


class Interaction(Module, ABC):
    """Any interaction between several entities that can be parameterized by phi / phi_t.

    Attributes:
        entities: the entities involved in the interaction.
    """
    entities: List[Entity]

    def __init__(self, entities: List[Entity]) -> None:
        super().__init__()
        self.entities = entities

    def get_configuration(self, i=-1) -> List[Tensor]:
        return utils.filter_none([entity.get_configuration(i) for entity in self.entities])

    def get_configuration_previous(self) -> List[Tensor]:
        return self.get_configuration(i=-2)

    def get_velocity(self, i=-1) -> List[Tensor]:
        return utils.filter_none([entity.get_velocity(i) for entity in self.entities])

    def get_velocity_previous(self) -> List[Tensor]:
        return self.get_velocity(i=-2)

    def get_control(self, i=-1) -> List[Tensor]:
        return utils.filter_none([entity.get_control(i) for entity in self.entities])

    def get_control_previous(self) -> List[Tensor]:
        return self.get_control(i=-2)


    @abstractmethod
    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        """Compute the normal distances between entities.

        For a basic polygon-ground interaction, phi would have one entry for every vertex in
        the polygon representing the distance betwen that polygon and the ground.

        Args:
            configurations: a list of batch_n x configuration_n x 1 tensors.

        Returns:
            A distance tensor of dimension batch_n x contact_n x 1.
        """
        pass

    def compute_phi_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_phi(configurations)

    def compute_phi_previous(self) -> Tensor:
        return self.compute_phi_history(i=-2)


    @abstractmethod
    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        """Compute the jacobian of phi w.r.t. configuration.

        Args:
            configurations: a list of batch_n x configuration_n x 1 tensors.

        Returns:
            A jacobian tensor of dimension batch_n x contact_n x configuration_n.
        """
        pass

    def compute_Jn_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_Jn(configurations)

    def compute_Jn_previous(self) -> Tensor:
        return self.compute_Jn_history(i=-2)


    @abstractmethod
    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        """Compute the tangential distances between entities.

        For a basic 2D polygon-ground interaction, phi would have one entry for every vertex in
        the polygon representing the distance horizontally for that vertex. For a 3D polygon-
        ground interaction, phi would have two entries for every vertex (horizontal x and
        horizontal y). These entries are all stacked interleaved in the second dimension of
        the tensor.

        Args:
            configurations: a list of batch_n x configuration_n x 1 tensors.

        Returns:
            A distance tensor of dimension batch_n x (r * contact_n) x 1. r represents the
            number of tangential distances for each contact. These are then interleaved. So if
            we are in 3D and have x and y distances, this returns along the second dimension
            vertex 1's x and y distance, followed by vertex 2's x and y distance, etc.
        """
        pass

    def compute_phi_t_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_phi_t(configurations)

    def compute_phi_t_previous(self) -> Tensor:
        return self.compute_phi_t_history(i=-2)


    @abstractmethod
    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        """Compute the jacobian of phi_t w.r.t. configuration.

        Args:
            configurations: a list of batch_n x configuration_n x 1 tensors.

        Returns:
            A jacobian tensor of dimension batch_n x (r * contact_n) x configuration_n.
            Interleaving is performed as described by compute_phi_t.
        """
        pass

    def compute_Jt_tilde_history(self, i=-1) -> Tensor:
        configurations = [entity.get_configuration(i) for entity in self.entities]
        return self.compute_Jt_tilde(configurations)

    def compute_Jt_tilde_previous(self) -> Tensor:
        return self.compute_Jt_tilde_history(i=-2)


    @abstractmethod
    def contact_n(self) -> int:
        """Number of contacts; the number of elements that will be returned by compute_phi."""
        pass


    def compute_M(self, configurations: List[Tensor]) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M(configuration)
                                        for entity, configuration in
                                        zip(self.entities, configurations)])

    def compute_M_history(self, i=-1) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M_history(i=i)
                                        for entity in self.entities])

    def compute_M_previous(self) -> Tensor:
        return self.compute_M_history(i=-2)


    def compute_M_i(self, configurations: List[Tensor]) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M_i(configuration)
                                        for entity, configuration in
                                        zip(self.entities, configurations)])

    def compute_M_i_history(self, i=-1) -> Tensor:
        return tensor_utils.block_diag([entity.compute_M_i_history(i=i)
                                        for entity in self.entities])

    def compute_M_i_previous(self) -> Tensor:
        return self.compute_M_i_history(i=-2)


    def compute_gamma(self, configurations: List[Tensor]) -> Tensor:
        return tensor_utils.block_diag([entity.compute_gamma(configuration)
                                        for entity, configuration in
                                        zip(self.entities, configurations)])

    def compute_gamma_history(self, i=-1) -> Tensor:
        return tensor_utils.block_diag([entity.compute_gamma_history(i=i)
                                        for entity in self.entities])

    def compute_gamma_previous(self) -> Tensor:
        return self.compute_gamma_history(i=-2)


    def compute_f(self, sp: 'SystemParams', configurations: List[Tensor],
                  velocities: List[Tensor], controls: List[Tensor], dt=None) -> Tensor:
        f_list = [entity.compute_f(sp, configuration, velocity, control, dt=dt)
                  for entity, configuration, velocity, control in
                  zip(self.entities, configurations, velocities, controls)]
        f_list = utils.filter_none(f_list)

        return torch.cat(f_list, dim=1)

    def compute_f_history(self, sp: 'SystemParams', i=-1, dt=None) -> Tensor:
        f_list = utils.filter_none([entity.compute_f_history(sp, i=i, dt=dt)
                                    for entity in self.entities])

        return torch.cat(f_list, dim=1)

    def compute_f_previous(self, sp: 'SystemParams') -> Tensor:
        return self.compute_f_history(sp, i=-2)


    def batch_n(self) -> int:
        batch_ns = [entity.batch_n() for entity in self.entities]
        assert(utils.elements_identical(batch_ns))
        return batch_ns[0]


class DirectInteraction(Module, ABC):
    """Any interaction between several entities that is parameterized directly.

    Used for the deep end-to-end baselines. Simply outputs impulses for each entity in the
    interaction.

    Attributes:
        entities: the entities involved in the interaction.
    """
    entities: List[Entity]

    def __init__(self, entities: List[Entity]) -> None:
        super().__init__()
        self.entities = entities

    @abstractmethod
    def compute_impulses(self, sp: 'SystemParams') -> List[Tensor]:
        pass
