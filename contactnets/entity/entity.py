import torch
from torch import Tensor, Size
from torch.nn import Module

from abc import ABC, abstractmethod
from typing import *

import pdb

class Entity(Module, ABC):
    # batch_n x n x 1 tensors, can be n=0 for stateless entity (e.g. ground)
    configuration_history: List[Tensor]
    velocity_history: List[Tensor]
    control_history: List[Tensor]

    configuration_length: int
    velocity_length: int

    def __init__(self, configuration_length: int, velocity_length: int) -> None:
        super(Entity, self).__init__()

        self.configuration_length = configuration_length
        self.velocity_length = velocity_length

    def check_consistent(self) -> None:
        #print(len(self.configuration_history),len(self.configuration_history),len(self.control_history))
        assert len(self.configuration_history) == \
               len(self.velocity_history) == \
               len(self.control_history)

    def set_state(self, configuration: Tensor, velocity: Tensor) -> None:
        self.clear_history()

        assert configuration.shape[1] == self.configuration_length
        assert velocity.shape[1] == self.velocity_length

        self.configuration_history.append(configuration)
        self.velocity_history.append(velocity)

    def get_state(self, i=-1) -> Tensor:
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

    def configuration_size(self) -> int:
        return self.configuration_length

    def velocity_size(self) -> int:
        return self.velocity_length

    def state_size(self) -> int:
        return self.configuration_size() + self.velocity_size()

    def append_state(self, configuration: Tensor, velocity: Tensor) -> None:
        self.configuration_history.append(configuration)
        self.velocity_history.append(velocity)

    def append_control(self, control: Tensor) -> None:
        self.control_history.append(control)

    def get_control(self, i=-1) -> Tensor:
        return self.control_history[i]

    def get_control_previous(self) -> Tensor:
        return self.get_control(i=-2)

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
    def compute_M(self, configuration: Tensor) -> Tensor:
        pass

    def compute_M_history(self, i=-1) -> Tensor:
        return self.compute_M(self.get_configuration(i))

    def compute_M_previous(self) -> Tensor:
        return self.compute_M_history(i=-2)


    @abstractmethod
    def compute_M_i(self, configuration: Tensor) -> Tensor:
        pass

    def compute_M_i_history(self, i=-1) -> Tensor:
        return self.compute_M_i(self.get_configuration(i))

    def compute_M_i_previous(self) -> Tensor:
        return self.compute_M_i_history(i=-2)


    @abstractmethod
    def compute_gamma(self, configuration: Tensor) -> Tensor:
        pass

    def compute_gamma_history(self, i=-1) -> Tensor:
        return self.compute_gamma(self.get_configuration(i))

    def compute_gamma_previous(self) -> Tensor:
        return self.compute_gamma_history(i=-2)


    @abstractmethod
    def compute_f(self, sp, configuration: Tensor, velocity: Tensor, control: Tensor,
                        dt: Tensor = None) -> Tensor:
        pass

    def compute_f_history(self, sp, i=-1, dt: Tensor = None) -> Tensor:
        return self.compute_f(sp, self.get_configuration(i), self.get_velocity(i),
                              self.get_control(i), dt=dt)

    def compute_f_previous(self, sp) -> Tensor:
        return self.compute_f_history(sp, i=-2)


    def undo_step(self) -> None:
        self.check_consistent()
        self.configuration_history.pop()
        self.velocity_history.pop()
        self.control_history.pop()

    def clear_history(self) -> None:
        self.configuration_history = []
        self.velocity_history = []
        self.control_history = []

    def reset_history(self) -> None:
        self.configuration_history = self.configuration_history[0:1]
        self.velocity_history = self.velocity_history[0:1]
        self.control_history = []
