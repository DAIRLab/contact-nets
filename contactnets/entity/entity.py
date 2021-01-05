from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from contactnets.system import SystemParams


class Entity(Module, ABC):
    """Any physical entity in the environment.

    For any of the histories, the end of the list is the most recent entry.

    Attributes:
        configuration_history: List[batch_n x configuration_n x 1].
        velocity_history: List[batch_n x velocity_n x 1].
        control_history: List[batch_n x velocity_n x 1].

        configuration_n: the size of the entity's configuration. For example a 2D
        rigid object can be represented by three variables (x, y, theta).

        velocity_n: the size of the entity's velocity. This can be different than the
        configuration length. For example, a 3D rigid object would have a configuration length
        of 7 (x, y, z, quaternion) but only a velocity length of 6 since an angular velocity
        only requires three variables.
    """

    configuration_history: List[Tensor]
    velocity_history: List[Tensor]
    control_history: List[Tensor]

    configuration_n: int
    velocity_n: int

    def __init__(self, configuration_n: int, velocity_n: int) -> None:
        super().__init__()

        self.configuration_history = []
        self.velocity_history = []
        self.control_history = []

        self.configuration_n = configuration_n
        self.velocity_n = velocity_n

    def check_consistent(self, state_ahead=False) -> None:
        """Ensure that all history lists are consistent.

        Args:
            state_ahead: if True, make sure that configuration and velocity history lengths are
            equal and one more than the control history length. Otherwise they should all be the
            same.
        """
        assert len(self.configuration_history) == \
               len(self.velocity_history) == \
               len(self.control_history) + (1 if state_ahead else 0)

    def get_state(self, i=-1) -> Tensor:
        """Return the state of the object as a Tensor of dimension:
            batch_n x (configuration_n + velocity_n) x 1."""
        return torch.cat((self.get_configuration(i),
                          self.get_velocity(i)), dim=1)

    def get_state_previous(self) -> Tensor:
        return self.get_state(i=-2)

    def get_configuration(self, i=-1) -> Tensor:
        return self.configuration_history[i]

    def get_configuration_previous(self) -> Tensor:
        return self.get_configuration(i=-2)

    def get_velocity(self, i=-1) -> Tensor:
        return self.velocity_history[i]

    def get_velocity_previous(self) -> Tensor:
        return self.get_velocity(i=-2)

    def get_control(self, i=-1) -> Tensor:
        return self.control_history[i]

    def get_control_previous(self) -> Tensor:
        return self.get_control(i=-2)


    def set_state(self, configuration: Tensor, velocity: Tensor) -> None:
        """Clear all history and initializes to the specified configuration and velocity."""
        self.clear_history()

        assert configuration.shape[1] == self.configuration_n
        assert velocity.shape[1] == self.velocity_n

        self.configuration_history.append(configuration)
        self.velocity_history.append(velocity)

    def append_state(self, configuration: Tensor, velocity: Tensor) -> None:
        self.configuration_history.append(configuration)
        self.velocity_history.append(velocity)

    def append_control(self, control: Tensor) -> None:
        self.control_history.append(control)


    def undo_step(self) -> None:
        """Remove the last element from every history list."""
        self.check_consistent()
        self.configuration_history.pop()
        self.velocity_history.pop()
        self.control_history.pop()

    def clear_history(self) -> None:
        """Clear all history."""
        self.configuration_history = []
        self.velocity_history = []
        self.control_history = []

    def reset_history(self) -> None:
        """Clear all history except the first configuration and velocity.

        This is useful if you want to restart a simulation with some change to the interactions.
        We completely clear control_history because system.sim expects an initial configuration
        and velocity but no control.
        """
        self.configuration_history = self.configuration_history[0:1]
        self.velocity_history = self.velocity_history[0:1]
        self.control_history = []


    def configuration_size(self) -> int:
        return self.configuration_n

    def velocity_size(self) -> int:
        return self.velocity_n

    def state_size(self) -> int:
        return self.configuration_size() + self.velocity_size()

    def control_size(self) -> int:
        return self.velocity_n  # velocity and control are same length

    def batch_n(self) -> int:
        assert self.step_n() > 0
        assert self.configuration_history[0].shape[0] == \
               self.velocity_history[0].shape[0] == \
               self.control_history[0].shape[0]

        return self.configuration_history[0].shape[0]

    def step_n(self) -> int:
        self.check_consistent()
        return len(self.configuration_history)


    @abstractmethod
    def compute_M(self, configuration: Tensor) -> Optional[Tensor]:
        """Compute the mass / inertia matrix of the entity.

        Args:
            configuration: batch_n x configuration_n x 1

        Returns:
            Either None or a mass tensor of shape batch_n x configuration_n x configuration_n.
            None should be returned if Entity's mass is ill defined (such as a Ground
            entity that can never be moved).
        """
        pass

    def compute_M_history(self, i=-1) -> Optional[Tensor]:
        return self.compute_M(self.get_configuration(i))

    def compute_M_previous(self) -> Optional[Tensor]:
        return self.compute_M_history(i=-2)


    @abstractmethod
    def compute_M_i(self, configuration: Tensor) -> Optional[Tensor]:
        """Compute the inverse of the mass matrix."""
        pass

    def compute_M_i_history(self, i=-1) -> Optional[Tensor]:
        return self.compute_M_i(self.get_configuration(i))

    def compute_M_i_previous(self) -> Optional[Tensor]:
        return self.compute_M_i_history(i=-2)


    @abstractmethod
    def compute_gamma(self, configuration: Tensor) -> Optional[Tensor]:
        """Compute the jacobian relating velocity and configuration.

        Args:
            configuration: batch_n x configuration_n x 1

        Returns:
            Either None or a jacobian of shape batch_n x configuration_n x velocity_n.
            For a simple 2D polygon this is simply the 3 x 3 identity matrix but for a 3D
            polygon it will be nontrivial since we need to relate length 4 quaternions with
            length 3 angular velocities.
        """
        pass

    def compute_gamma_history(self, i=-1) -> Optional[Tensor]:
        return self.compute_gamma(self.get_configuration(i))

    def compute_gamma_previous(self) -> Optional[Tensor]:
        return self.compute_gamma_history(i=-2)


    @abstractmethod
    def compute_f(self, sp: 'SystemParams', configuration: Tensor, velocity: Tensor,
                  control: Tensor, dt: Tensor = None) -> Tensor:
        """Compute the contact free velocity after a dt time step. If dt is None then
        the dt attribute on sp should be used.

        Args:
            sp: The system parameters.

            configuration: batch_n x configuration_n x 1
            velocity: batch_n x velocity_n x 1
            control: batch_n x velocity_n x 1

            dt: batch_n x 1 x 1 tensor or None

        Returns:
            The batch_n x velocity_n x 1 contact free velocity.
        """
        pass

    def compute_f_history(self, sp: 'SystemParams', i=-1, dt: Tensor = None) -> Tensor:
        return self.compute_f(sp, self.get_configuration(i), self.get_velocity(i),
                              self.get_control(i), dt=dt)

    def compute_f_previous(self, sp: 'SystemParams') -> Tensor:
        return self.compute_f_history(sp, i=-2)
