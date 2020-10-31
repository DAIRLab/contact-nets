import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from typing import *
from dataclasses import dataclass, field

from contactnets.entity import Entity
from contactnets.interaction import InteractionResolver
from contactnets.utils import utils

import pdb

@dataclass
class SystemParams:
    dt: Tensor = field(default_factory=lambda: torch.tensor(0.1))
    g: Tensor = field(default_factory=lambda: torch.tensor(10.0))

@dataclass
class SimResult:
    times: List[float]
    configurations: List[List[Tensor]]
    velocities: List[List[Tensor]]

class System(Module):
    entities: ModuleList
    resolver: InteractionResolver
    params: SystemParams

    def __init__(self, entities: List[Entity],
                       resolver: InteractionResolver,
                       params: SystemParams) -> None:
        super().__init__()

        self.entities = ModuleList(entities)
        self.resolver = resolver
        self.params = params

    def batch_n(self) -> int:
        batch_ns = [entity.batch_n() for entity in self.entities]
        assert utils.elements_identical(batch_ns)
        return batch_ns[0]

    def step_n(self) -> int:
        step_ns = [entity.step_n() for entity in self.entities]
        assert utils.elements_identical(step_ns)
        return step_ns[0]

    def set_states(self, configurations: List[Tensor], velocities: List[Tensor]) -> None:
        assert len(self.entities) == len(configurations) == len(velocities)
        for entity, configuration, velocity in zip(self.entities, configurations, velocities):
            entity.set_state(configuration, velocity)

    def get_states(self) -> Tuple[List[Tensor], List[Tensor]]:
        configurations = []
        velocities = []

        for entity in self.entities:
            configurations.append(entity.get_configuration())
            velocities.append(entity.get_velocity())

        return configurations, velocities

    def configuration_histories(self) -> List[List[Tensor]]:
        config_histories = [entity.configuration_history for entity in self.entities]
        return utils.transpose_lists(config_histories)

    def velocity_histories(self) -> List[List[Tensor]]:
        velocity_histories = [entity.velocity_history for entity in self.entities]
        return utils.transpose_lists(velocity_histories)

    def control_histories(self) -> List[List[Tensor]]:
        control_histories = [entity.control_history for entity in self.entities]
        return utils.transpose_lists(control_histories)

    def append_controls(self, controls: List[Tensor]) -> None:
        assert len(self.entities) == len(controls)
        for entity, control in zip(self.entities, controls):
            entity.append_control(control)

    def step(self) -> None:
        for entity in self.entities:
            entity.check_consistent()

        self.resolver.step(self)

    def undo_step(self) -> None:
        for entity in self.entities:
            entity.undo_step()

    def sim(self, controls: List[List[Tensor]]) -> SimResult:
        #pdb.set_trace()
        for i in range(len(controls) - 1):
            #print(i)
            self.append_controls(controls[i])
            self.step()

        self.append_controls(controls[-1])

        return self.get_sim_result()

    def get_sim_result(self) -> SimResult:
        times = [x * self.params.dt.item() for x in range(self.step_n())]
        configurations = []
        velocities = []

        for entity in self.entities:
            configurations.append(entity.configuration_history)
            velocities.append(entity.velocity_history)

        return SimResult(times, configurations, velocities)

    def restart_sim(self) -> SimResult:
        control_histories = self.control_histories()

        for entity in self.entities:
            entity.reset_history()

        return self.sim(control_histories)

    def clear(self) -> None:
        for entity in self.entities:
            entity.clear_history()
