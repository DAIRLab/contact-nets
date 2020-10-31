import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from typing import *

from abc import ABC, abstractmethod

from contactnets.entity import Entity, Dynamic3D

from contactnets.utils import utils, tensor_utils

import pdb


class InteractionResolver(Module, ABC):
    interactions: ModuleList
    direct_interactions: ModuleList

    def __init__(self, interactions: ModuleList,
                       direct_interactions: ModuleList) -> None:
        super().__init__()

        self.interactions = interactions
        self.direct_interactions = direct_interactions

    def step_impulses(self, system: 'System', impulses: List[Tensor]) -> None:
        # Basic step handler that just steps using accumulated contact impulses
        # and basic physics (dx = dt * gamma * (impulse + gravity + control)
        # Might not be appropriate for more complex schemes (e.g. with elasticity)
        # Takes a list of batch_n x n x 1 impulse tensor, one per entity
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
                # Normalize quaternions each simulation step
                quat_norm = configuration_n[:,3:7,:].norm(dim = 1)
                configuration_n[:,3:7,:] /= quat_norm

            entity.append_state(configuration_n, velocity_n)

    @abstractmethod
    def step(self, system: 'System') -> None:
        pass
