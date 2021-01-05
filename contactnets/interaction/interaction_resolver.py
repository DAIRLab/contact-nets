from abc import ABC, abstractmethod
import pdb  # noqa
from typing import TYPE_CHECKING, List

from torch import Tensor
from torch.nn import Module, ModuleList

from contactnets.entity import Dynamic3D

if TYPE_CHECKING:
    from contactnets.system import System


class InteractionResolver(Module, ABC):
    """A base class that describes how to resolve several interactions.

    Attributes:
        interactions: all the interactions that parameterize with phi / phi_t.
        direct_interactions: all the interactions that directly output impulses.
    """
    interactions: ModuleList
    direct_interactions: ModuleList

    def __init__(self, interactions: List[Module], direct_interactions: List[Module]) -> None:
        super().__init__()

        self.interactions = ModuleList(interactions)
        self.direct_interactions = ModuleList(direct_interactions)

    def step_impulses(self, system: 'System', impulses: List[Tensor]) -> None:
        """Update states of system entities using specified contact impulses and basic physics.

        These impulses can come from either a regular or direct interaction. The child class
        would then call this method within its overriden step function. Note that more complex
        resolvers such as the ElasticLCP might not use this.

        Args:
            system: the system to update.
            impulses: A list of batch_n x velocity_n x 1 impulse tensors, one per entity.
        """

        sp = system.params

        for entity, impulse in zip(system.entities, impulses):
            M_i = entity.compute_M_i_history()
            gamma = entity.compute_gamma_history()

            velocity_n = entity.compute_f_history(sp).clone()
            if M_i is not None:
                velocity_n += M_i.bmm(impulse)

            configuration_n = entity.get_configuration().clone()
            if gamma is not None:
                configuration_n += sp.dt * gamma.bmm(velocity_n)

            if isinstance(entity, Dynamic3D):
                # Normalize quaternions each simulation step for numerical stability
                quat_norm = configuration_n[:, 3:7, :].norm(dim = 1)
                configuration_n[:, 3:7, :] /= quat_norm

            entity.append_state(configuration_n, velocity_n)

    @abstractmethod
    def step(self, system: 'System') -> None:
        """Simulates the system forward one time step.

        Before this function is called, all entities will have configuration, velocity, and
        control histories of length n. Afterwards, the configuration and velocity histories of
        all entities should be length n + 1, with control still at length n. The system will
        then append the control input to all entities and call step() again.

        Args:
            system: the system to update
        """
        pass
