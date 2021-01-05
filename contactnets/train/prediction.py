from enum import Enum
import pdb  # noqa
from typing import List

import torch
from torch import Tensor

from contactnets.system import System
from contactnets.train import Loss, LossType
from contactnets.utils import quaternion


class PredictionMode(Enum):
    XY = 1
    XY_ROT = 2
    XYZ_QUAT = 3


def get_names(mode: PredictionMode):
    names = []
    if mode == PredictionMode.XY: names = ['pos_pred', 'vel_pred']
    elif mode == PredictionMode.XY_ROT or PredictionMode.XYZ_QUAT:
        names = ['pos_pred', 'angle_pred', 'vel_pred', 'angle_vel_pred']
    return names


class PredictionLoss(Loss):
    """Compute prediction error over one time step."""
    mode: PredictionMode
    norm: int

    def __init__(self, system: System, mode: PredictionMode, norm = 2) -> None:
        super().__init__(get_names(mode), system, LossType.STEPWISE, True)
        self.mode = mode
        self.norm = norm

    def compute_loss(self, meta_save_file: str = None) -> List[Tensor]:
        configurations_data, velocities_data = self.system.get_states()
        self.system.undo_step()
        self.system.step()
        configurations_sim, velocities_sim = self.system.get_states()
        return self.compute_loss_from_states(configurations_data, velocities_data,
                                             configurations_sim,  velocities_sim)

    def compute_loss_from_states(self, configurations_data: List[Tensor],
                                 velocities_data: List[Tensor],
                                 configurations_sim: List[Tensor],
                                 velocities_sim: List[Tensor]) -> List[Tensor]:
        if self.mode == PredictionMode.XY:
            pos_diff, vel_diff = torch.tensor(0.0), torch.tensor(0.0)
            for data, sim in zip(configurations_data, configurations_sim):
                pos_diff += torch.norm(data - sim, self.norm)

            for data, sim in zip(velocities_data, velocities_sim):
                vel_diff += torch.norm(data - sim, self.norm)
            return [pos_diff, vel_diff]

        if self.mode == PredictionMode.XY_ROT:
            pos_diff, angle_diff, vel_diff, angle_vel_diff = \
                torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

            for data, sim in zip(configurations_data, configurations_sim):
                if data.numel() > 0:
                    diff = data - sim
                    pos_diff += torch.norm(diff[:, 0:2, :], self.norm)
                    angle_diff += torch.norm(diff[:, 2, :], self.norm)

            for data, sim in zip(velocities_data, velocities_sim):
                if data.numel() > 0:
                    diff = data - sim
                    vel_diff += torch.norm(diff[:, 0:2, :], self.norm)
                    angle_vel_diff += torch.norm(diff[:, 2, :], self.norm)

            return [pos_diff, angle_diff, vel_diff, angle_vel_diff]

        if self.mode == PredictionMode.XYZ_QUAT:
            # TODO: implement angles
            pos_diff, angle_diff, vel_diff, angle_vel_diff = \
                torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            for data, sim in zip(configurations_data, configurations_sim):
                if data.numel() > 0:
                    diff = data - sim
                    pos_diff += torch.norm(diff[:, 0:3, :], self.norm)
                    angle_diff += quaternion.qdiff(data[:, 3:7, :].squeeze(2),
                                                   sim[:, 3:7, :].squeeze(2))[0]

            for data, sim in zip(velocities_data, velocities_sim):
                if data.numel() > 0:
                    diff = data - sim
                    vel_diff += torch.norm(diff[:, 0:3, :], self.norm)
                    angle_vel_diff += torch.norm(diff[:, 3:6, :], self.norm)

            return [pos_diff, angle_diff, vel_diff, angle_vel_diff]

        raise Exception('Prediction loss mode not recognized')


class BasicVelocityLoss(Loss):
    norm: int

    def __init__(self, system: System, norm = 2) -> None:
        super().__init__(['vel_basic'], system, LossType.STEPWISE, True)
        self.norm = norm

    def compute_loss(self, meta_save_file: str = None) -> List[Tensor]:
        configurations_data, velocities_data = self.system.get_states()
        self.system.undo_step()
        self.system.step()
        configurations_sim, velocities_sim = self.system.get_states()

        velocities_diff = torch.tensor(0.0)
        for data, sim in zip(velocities_data, velocities_sim):
            velocities_diff += torch.norm(data - sim, self.norm)

        return [velocities_diff]
