from dataclasses import dataclass
import json
from numbers import Number
import os
import os.path
import pdb  # noqa
from typing import Dict

import click
import numpy as np
import skvideo.io
from tensorboardX import SummaryWriter
import torch
from torch import Tensor
from torch.optim.lr_scheduler import StepLR

from contactnets.experiments.ballbin2d import BallBin2DParams, PairwiseStructuredLearnable, sim
from contactnets.system import System
from contactnets.train import (BallBin2DSurrogate, BinSurrogateConfig2D, DataManager,
                               LossManager, PredictionLoss, PredictionMode, ReportingLoss,
                               TensorboardManager, TensorboardPlot, Trainer, TrajectoryLoss)
from contactnets.utils import dirs, file_utils, system_io, tensor_utils
from contactnets.vis import Visualizer2D

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def save_toss_video(writer: SummaryWriter, system: System, bp: BallBin2DParams,
                    run: Tensor, name: str) -> None:
    system_io.load_system(run, system)
    sim_results = [system.get_sim_result()]
    system.restart_sim()
    sim_results.append(system.get_sim_result())

    lcp = system.resolver

    vis = Visualizer2D(sim_results, lcp.interactions[0].ball_geometries,
                       lcp.interactions[0].walls, system.params)
    vis.camera.y = 4
    vis.camera.zoom = 40.0
    file_path = vis.render(save_video=True)

    if file_path is None: return

    video = skvideo.io.vread(file_path)
    # batch_n = 1 x time_steps x colors x H x W
    # Move color and add empty batch dim
    video = np.expand_dims(np.moveaxis(video, 3, 1), 0)
    # Downsample video
    video = video[:, :, :, 3::4, 3::4]

    writer.add_video(name, video, fps=15)

    os.remove(file_path)


ct = 'Multiline'
structured_custom_scalars = {
    'Trajectory': {
        'Pos Int (percent)':  [ct, ['train_pos_int_traj',         'valid_pos_int_traj']],
        'Angle Int (degree)': [ct, ['train_angle_int_traj',       'valid_angle_int_traj']],
        'Pen Int (percent)':  [ct, ['train_penetration_int_traj', 'valid_penetration_int_traj']]
    },
    'Surrogate': {'Surrogate': [ct, ['train_surrogate',            'valid_surrogate']]},
    'Time':      {'Time':      [ct, ['time_train',                 'time_log']]}
}


# FIELDS MUST BE DECORATED IN ORDER FOR DATACLASS TO WORK
@dataclass
class BallBin2DTraining:
    learn_normal: bool = True
    learn_tangent: bool = True

    polytope: bool = True

    surrogate_config: BinSurrogateConfig2D = BinSurrogateConfig2D(
        w_comp_n                = 1.0,
        w_comp_t                = 1.0,
        w_match                 = 1.0,
        w_cone                  = 1.0,
        w_penetration_slack     = 1.0,

        w_penetration           = 1.0,
        w_config_grad_normal    = 0.3,
        w_config_grad_tangent   = 0.0,
        w_config_grad_perp      = 0.3,
        w_st_estimate_pen       = 0.0,
        w_st_estimate_normal    = 0.0,
        w_st_estimate_tangent   = 0.0,
        w_tangent_jac_d2        = 0.0,
        w_contact_threshold     = 0.0
    )

    robust_sqrt:            bool  = True

    device: str = 'cpu'
    lr: float = 5e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 0.0
    noise: float = 0.0
    H: int = 256

    epochs: int = 0
    batch: int = 0


def do_train_structured(epochs: int, batch: int, patience: int, resume: bool,
                        training = BallBin2DTraining(), video: bool = True):
    training.epochs = epochs
    training.batch = batch

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    tensor_utils.set_default_tensor_type(device)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)

    bp: BallBin2DParams = file_utils.load_params(device, 'experiment')

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp)
    balls = system.resolver.interactions[0].balls
    walls = system.resolver.interactions[0].walls
    interaction = PairwiseStructuredLearnable(balls, walls, bp.radii, bp.ball_mus, bp.wall_mus)
    ground_truth_phi = system.resolver.interactions[0].compute_phi
    system.resolver.interactions = torch.nn.ModuleList([interaction])

    # Formulate the training loss, as well as reporting losses
    loss = BallBin2DSurrogate(system, interaction, config=training.surrogate_config)

    prediction_loss = PredictionLoss(system, mode=PredictionMode.XY_ROT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss, ground_truth_phi=ground_truth_phi)

    reporting_losses = [ReportingLoss(loss, 1000),
                        ReportingLoss(prediction_loss, 1000),
                        ReportingLoss(trajectory_loss, 15)]

    loss_manager = LossManager(system, [loss], reporting_losses)

    # Specify the optimizer to use
    def optimizer_factory(params):
        return torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)

    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

    def tensorboard_callback(writer: SummaryWriter, epoch_vars: Dict[str, Number]) -> None:
        if epoch_vars['epoch'] == 0:
            git_file = dirs.out_path('git.txt')
            if os.path.exists(git_file):
                writer.add_text('git', file_utils.read_file(git_file).replace('\n', '  \n'))

            training_text = json.dumps(training.__dict__, sort_keys=True, indent=4,
                                       default=lambda x: x.__dict__).replace('\n', '  \n')
            experiment_text = json.dumps(file_utils.listify_tensors(bp).__dict__,
                                         sort_keys=True, indent=4).replace('\n', '  \n')
            writer.add_text('training', training_text)
            writer.add_text('experiment', experiment_text)

            writer.add_custom_scalars(structured_custom_scalars)

        if video:
            for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
                save_toss_video(writer, system, bp, data_manager.train_runs[i],
                                f'f_toss(red=data,blue=prediciton)/train{i}')
                save_toss_video(writer, system, bp, data_manager.valid_runs[i],
                                f'f_toss(red=data,blue=prediciton)/valid{i}')

    # Set up tensorboard logging
    loss_names = ['train_' + name for name in loss.names]
    plot_vars = [loss_names[0], 'valid_surrogate'] + loss_names[1:]
    plot = TensorboardPlot('a_loss', plot_vars, [], True)
    tensorboard = TensorboardManager(callback=tensorboard_callback)
    tensorboard.plots.append(plot)


    epoch_prints = [('train_surrogate', 'train_sur'),
                    ('valid_surrogate', 'valid_sur'),
                    ('train_vel_pred',  'train_l2_vel'),
                    ('valid_vel_pred',  'valid_l2_vel')]

    # Create the trainer
    trainer = Trainer(system, data_manager, loss_manager, tensorboard,
                      epoch_prints=epoch_prints,
                      optimizer_factory=optimizer_factory,
                      scheduler_factory=scheduler_factory)

    file_utils.save_params(training, 'training')

    file_utils.create_empty_directory(dirs.out_path('renders'))
    tensorboard.create_writer(resume=resume)
    tensorboard.launch()

    if resume: trainer.load_training()

    trainer.train(epochs=epochs, batch_size=batch, evaluation_loss='valid_surrogate',
                  patience=patience, surpress_printing=True)


@click.command()
@click.option('--epochs', default=100)
@click.option('--batch', default=1)
@click.option('--patience', type=int, default=None)
@click.option('--resume/--restart', default=False)
@click.option('--resume/--restart', default=False)
@click.option('--video/--no-video', default=True)
def main(epochs: int, batch: int, patience: int, resume: bool, video: bool):
    do_train_structured(epochs, batch, patience, resume, video=video)


if __name__ == "__main__": main()
