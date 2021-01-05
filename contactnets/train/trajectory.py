import pdb  # noqa
from typing import List

import torch
from torch import Tensor

from contactnets.system import System
from contactnets.train import Loss, LossType, PredictionLoss, PredictionMode
from contactnets.utils import quaternion, system_io


def get_names(prediction: PredictionLoss, chop_fraction: float):
    chopstr = str(round(chop_fraction * 100))
    names = ['pos_final_traj', 'angle_final_traj',
             'pos_int_traj', 'angle_int_traj',
             'pos_bound_traj', 'angle_bound_traj',
             'penetration_int_traj', 'penetration_max_traj']
    if round(chop_fraction * 100) > 0:
        for i in range(len(names)):
            names[i] += '_' + chopstr
    return names


class TrajectoryLoss(Loss):
    """Evaluate a prediction loss over a trajectory.

    Attributes:
        prediction: the prediction loss to evaluate.

        pos_error_bound: pos_bound_traj returns for how many time steps the simulated trajectory
        position was within a tolerance of the ground truth trajectory. pos_error_bound is that
        tolerance.

        angle_error_bound: angle_bound_traj returns for how many time steps the simulated
        trajectory angle was within a tolerance of the ground truth trajectory.
        angle_error_bound is that tolerance.
    """
    prediction: PredictionLoss
    pos_error_bound: float
    angle_error_bound: float

    def __init__(self, system: System, prediction: PredictionLoss, chop_fraction = 0.0,
                 ground_truth_phi = lambda x: torch.tensor(0)) -> None:
        super().__init__(get_names(prediction, chop_fraction), system,
                         LossType.TRAJECTORY, True)
        self.prediction = prediction
        self.pos_error_bound = 0.1
        self.angle_error_bound = 0.1
        self.chop_fraction = chop_fraction
        self.ground_truth_phi = ground_truth_phi

    def preprocess_data(self, data):
        trajlen = data.shape[1]
        start = round(trajlen * self.chop_fraction)
        return data[:, start:, :]

    def compute_travel_dist(self, config_hist: List[List[Tensor]], pos: bool) -> Tensor:
        if self.prediction.mode in [PredictionMode.XY, PredictionMode.XY_ROT]:
            # TODO: not quite right for multiple objects
            dynamic_configs_list = [torch.cat(config, dim=1) for config in config_hist]
            dynamic_configs = torch.cat(dynamic_configs_list, dim=2)

            rot_inds = torch.arange(0, dynamic_configs.shape[1]) % 3 == 2
            pos_inds = rot_inds == False  # noqa

            configs_sel = dynamic_configs[:, pos_inds if pos else rot_inds, :]

            configs_deltas = configs_sel[:, :, 1:] - configs_sel[:, :, :-1]
            configs_deltas = configs_deltas.norm(dim=1, p=2)
            return torch.sum(configs_deltas, dim=1).unsqueeze(1)
        else:
            return torch.tensor([1.])

    def compute_loss(self, meta_save_file: str = None) -> List[Tensor]:
        assert self.system.batch_n() == 1

        config_hist_data = self.system.configuration_histories()
        vel_hist_data = self.system.velocity_histories()
        self.system.restart_sim()
        config_hist_sim = self.system.configuration_histories()
        vel_hist_sim = self.system.velocity_histories()

        if meta_save_file is not None:
            traj = system_io.serialize_system(self.system)
            torch.save(traj, meta_save_file)

        state_losses = []
        max_penetration, sum_penetration = torch.tensor(0.), torch.tensor(0.)
        for config_data, vel_data, config_sim, vel_sim in \
                zip(config_hist_data, vel_hist_data, config_hist_sim, vel_hist_sim):
            state_loss = self.prediction.compute_loss_from_states(
                config_data, vel_data, config_sim, vel_sim)
            state_losses.append(state_loss)
            pen_i = self.ground_truth_phi(config_sim).min()
            if pen_i < max_penetration:
                max_penetration = pen_i
            if pen_i < 0:
                sum_penetration += pen_i
        sum_penetration /= -len(config_hist_data)
        max_penetration *= -1

        pos_travel_dist = self.compute_travel_dist(config_hist_data, pos=True)
        ang_travel_dist = self.compute_travel_dist(config_hist_data, pos=False)

        final_config_data = config_hist_data[-1][0]
        final_config_sim = config_hist_sim[-1][0]
        final_config_diff = final_config_data - final_config_sim
        if self.prediction.mode in [PredictionMode.XY, PredictionMode.XY_ROT]:
            final_pos_diff = final_config_diff[:, 0:2, :].norm(dim=1, p=self.prediction.norm)
            final_ang_diff = final_config_diff[:, 2:3, :].norm(dim=1, p=self.prediction.norm)
        elif self.prediction.mode is PredictionMode.XYZ_QUAT:
            final_pos_diff = final_config_diff[:, 0:3, :].norm(dim=1, p=self.prediction.norm)
            final_ang_diff = quaternion.qdiff(final_config_data[:, 3:7, :].squeeze(2),
                                              final_config_sim[:, 3:7, :].squeeze(2))

        pos_error_percent = final_pos_diff / pos_travel_dist
        ang_error_percent = final_ang_diff / ang_travel_dist

        pos_sum = torch.tensor(0.0)
        angle_sum = torch.tensor(0.0)

        pos_in_bound_n = 0
        angle_in_bound_n = 0

        for i, state_loss in enumerate(state_losses):
            pos_sum += state_loss[0]
            angle_sum += state_loss[1]

            if state_loss[0] < self.pos_error_bound:
                pos_in_bound_n = i

            if state_loss[1] < self.angle_error_bound:
                angle_in_bound_n = i

        pos_sum /= len(config_hist_data)
        angle_sum /= len(config_hist_data)

        pos_sum *= 100 / 2.0
        angle_sum *= 57.3
        sum_penetration *= 100 / 2.0
        max_penetration *= 100 / 2.0

        return [pos_error_percent.sum(), ang_error_percent.sum(), pos_sum,
                angle_sum, torch.tensor(pos_in_bound_n), torch.tensor(angle_in_bound_n),
                sum_penetration, max_penetration]
