import pdb  # noqa

import torch
from torch import Tensor

from contactnets.system import System


def data_len(system: System) -> int:
    """Compute number of entries required to serialize all entities in a system."""
    entity_lens = [entity.state_size() + entity.control_size() for entity in system.entities]
    return sum(entity_lens)


def load_system(x: Tensor, system: System) -> None:
    """Loads the data tensor into the system entities.

    Args:
        x: batch_n x step_n x ((configuration_n + velocity_n + control_n) * entity_n)
        Note that configuration_n / velocity_n / control_n can vary for each entity so this
        formula is a bit imprecise.

        system: the target system.
    """
    step_n = x.shape[1]
    system.clear()

    for step in range(step_n):
        i = 0
        for entity in system.entities:
            configuration = x[:, step:step + 1, i:i + entity.configuration_size()]
            configuration = configuration.transpose(1, 2)
            i += entity.configuration_size()

            velocity = x[:, step:step + 1, i:i + entity.velocity_size()].transpose(1, 2)
            i += entity.velocity_size()

            control = x[:, step:step + 1, i:i + entity.velocity_size()].transpose(1, 2)
            i += entity.velocity_size()

            entity.append_state(configuration, velocity)
            entity.append_control(control)


def serialize_system(system: System) -> Tensor:
    """Serializes a system into a data tensor.

    Args:
        system: the system to serialize.

    Returns:
        batch_n x step_n x ((configuration_n + velocity_n + control_n) * entity_n)
        Note that configuration_n / velocity_n / control_n can vary for each entity so this
        formula is a bit imprecise.
    """
    batch_n, step_n, data_n = system.batch_n(), system.step_n(), data_len(system)

    x = torch.zeros(batch_n, step_n, data_n)

    for step in range(step_n):
        i = 0
        for entity in system.entities:
            x[:, step:step + 1, i:i + entity.configuration_size()] = \
                entity.configuration_history[step].transpose(1, 2)
            i += entity.configuration_size()

            x[:, step:step + 1, i:i + entity.velocity_size()] = \
                entity.velocity_history[step].transpose(1, 2)
            i += entity.velocity_size()

            x[:, step:step + 1, i:i + entity.velocity_size()] = \
                entity.control_history[step].transpose(1, 2)
            i += entity.velocity_size()
        assert i == data_n

    return x
