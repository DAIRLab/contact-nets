import torch
from torch import Tensor
from torch.nn import Module

from typing import *

from contactnets.system import System

import pdb

def data_len(system: System) -> int:
    data_len = 0
    for entity in system.entities:
        data_len += entity.configuration_size() + entity.velocity_size() * 2

    return data_len

def get_controls(x: Tensor, system: System) -> List[List[Tensor]]:
    step_n = x.shape[1]
    controls = []
    for step in range(step_n):
        i = 0
        step_controls = []
        for entity in system.entities:
            i = i + entity.configuration_size() + entity.velocity_size()
            control = x[:, step:step+1, i:i+entity.velocity_size()].transpose(1,2)
            i = i + entity.velocity_size()
            step_controls.append(control)
        controls.append(step_controls)
    return controls

def load_system(x: Tensor, system: System) -> None:
    # x is of dimension batch_n x step_n x [configuration velocity control]s
    step_n = x.shape[1]
    system.clear()

    for step in range(step_n):
        i = 0
        for entity in system.entities:
            configuration = x[:, step:step+1, i:i+entity.configuration_size()].transpose(1,2) 
            i = i + entity.configuration_size()

            velocity = x[:, step:step+1, i:i+entity.velocity_size()].transpose(1,2) 
            i = i + entity.velocity_size()

            control = x[:, step:step+1, i:i+entity.velocity_size()].transpose(1,2)
            i = i + entity.velocity_size()

            entity.append_state(configuration, velocity)
            entity.append_control(control)

def serialize_system(system: System) -> Tensor:
    batch_n = system.batch_n()
    step_n = system.step_n()
    data_n = data_len(system)

    x = torch.zeros(batch_n, step_n, data_n)

    for step in range(step_n):
        i = 0
        for entity in system.entities:
            x[:, step:step+1, i:i+entity.configuration_size()] = entity.configuration_history[step].transpose(1,2)
            i = i+entity.configuration_size() 

            x[:, step:step+1, i:i+entity.velocity_size()] = entity.velocity_history[step].transpose(1,2) 
            i = i+entity.velocity_size() 

            x[:, step:step+1, i:i+entity.velocity_size()] = entity.control_history[step].transpose(1,2) 
            i = i+entity.velocity_size() 
        assert(i == data_n)
    
    return x
