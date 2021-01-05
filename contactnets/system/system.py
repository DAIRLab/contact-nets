from dataclasses import dataclass, field
import pdb  # noqa
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from contactnets.entity import Entity
from contactnets.interaction import InteractionResolver
from contactnets.utils import utils


@dataclass
class SystemParams:
    """The parameters necessary to simulate a system."""
    dt: Tensor = field(default_factory=lambda: torch.tensor(0.1))
    g: Tensor = field(default_factory=lambda: torch.tensor(10.0))


@dataclass
class SimResult:
    """The result of a simulation.

    Attributes:
        times: a list of step_n times for the discrete simulation. So if our time step is 1
        second and we simulate 5 time steps, times would be [0.0, 1.0, 2.0, 3.0, 4.0, 5.0].

        configurations: a list of configuration histories for each entity. So configurations[0]
        would be a step_n length list of configurations for the first entity in the system.

        velocities: a list of velocity histories for each entity. So velocities[0] would be a
        step_n length list of velocities for the first entity in the system.
    """
    times: List[float]
    configurations: List[List[Tensor]]
    velocities: List[List[Tensor]]


class System(Module):
    """A physical system that can be simulated.

    Since the entities attribute is a ModuleList and resolver is a Module itself that contains
    its interactions as a ModuleList, calling system.parameters() will aggregate all the
    trainable parameters from both entities and interactions.

    Attributes:
        entities: all the entities in the system.
        resolver: one resolver that manages its interactions and the entities in the system.
        params: global parameters for simulation.
    """
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

    def check_consistent(self, state_ahead=False):
        for entity in self.entities:
            entity.check_consistent(state_ahead=state_ahead)

    def set_states(self, configurations: List[Tensor], velocities: List[Tensor]) -> None:
        assert len(self.entities) == len(configurations) == len(velocities)
        for entity, configuration, velocity in zip(self.entities, configurations, velocities):
            entity.set_state(configuration, velocity)

    def get_states(self) -> Tuple[List[Tensor], List[Tensor]]:
        configurations = [entity.get_configuration() for entity in self.entities]
        velocities = [entity.get_velocity() for entity in self.entities]

        return configurations, velocities

    def configuration_histories(self) -> List[List[Tensor]]:
        """Return a step_n length list where a sublist contains each entity's configuration."""
        config_histories = [entity.configuration_history for entity in self.entities]
        return utils.transpose_lists(config_histories)

    def velocity_histories(self) -> List[List[Tensor]]:
        """Return a step_n length list where a sublist contains each entity's velocity."""
        velocity_histories = [entity.velocity_history for entity in self.entities]
        return utils.transpose_lists(velocity_histories)

    def control_histories(self) -> List[List[Tensor]]:
        """Return a step_n length list where a sublist contains each entity's control."""
        control_histories = [entity.control_history for entity in self.entities]
        return utils.transpose_lists(control_histories)

    def append_controls(self, controls: List[Tensor]) -> None:
        assert len(self.entities) == len(controls)
        for entity, control in zip(self.entities, controls):
            entity.append_control(control)

    def step(self) -> None:
        self.check_consistent()
        self.resolver.step(self)

    def undo_step(self) -> None:
        for entity in self.entities:
            entity.undo_step()

    def sim(self, controls: List[List[Tensor]]) -> SimResult:
        """Forward simulate the system given control inputs.

        Each entity is expected to have its configuration and velocity history lengths at 1 and
        its control history length at zero.

        Args:
            controls: a list of length step_n where each element is a sublist containing the
            control inputs for each entity.

        Returns:
            The result of the simulation.
        """

        for i in range(len(controls) - 1):
            self.check_consistent(state_ahead=True)
            self.append_controls(controls[i])
            self.step()

        self.append_controls(controls[-1])

        return self.get_sim_result()

    def restart_sim(self) -> SimResult:
        """Restarts the simulation with the same controls and the same initial state.

        This could be useful if you want to modify an interaction and rerun the simulation to
        compare the final states.

        Returns:
            The result of the simulation.
        """
        control_histories = self.control_histories()

        for entity in self.entities:
            entity.reset_history()

        return self.sim(control_histories)

    def get_sim_result(self) -> SimResult:
        times = [x * self.params.dt.item() for x in range(self.step_n())]
        configurations = [entity.configuration_history for entity in self.entities]
        velocities = [entity.velocity_history for entity in self.entities]

        return SimResult(times, configurations, velocities)

    def clear(self) -> None:
        for entity in self.entities:
            entity.clear_history()
