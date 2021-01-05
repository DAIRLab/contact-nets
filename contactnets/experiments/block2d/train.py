# Ignore whitespace formatting for the custom scalars

from dataclasses import dataclass
from numbers import Number
import os
import os.path
import sys
from typing import Dict, cast

import click
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from contactnets.experiments.block2d import (Block2DParams, DeepLearnable, StructuredLearnable,
                                             sim)
from contactnets.experiments.block2d.train_utils import (tensorboard_callback_e2e,
                                                         tensorboard_callback_structured)
from contactnets.interaction import DirectResolver
from contactnets.train import (BasicVelocityLoss, DataManager, LossManager,
                               PolyGround2DSurrogate, PredictionLoss, PredictionMode,
                               ReportingLoss, SurrogateConfig2D, TensorboardManager,
                               TensorboardPlot, TensorboardSequentialHistogram, Trainer,
                               TrajectoryLoss)
from contactnets.utils import dirs, file_utils, tensor_utils

if __name__ == "__main__":
    # Make pygame renders headless
    os.environ['SDL_VIDEODRIVER'] = 'dummy'


# FIELDS MUST BE DECORATED IN ORDER FOR DATACLASS TO WORK
@dataclass
class Block2DTraining:
    learn_normal: bool = True
    learn_tangent: bool = True

    # 'poly' or 'deep'
    net_type: str = 'poly'

    surrogate_config: SurrogateConfig2D = SurrogateConfig2D(
        w_penetration          = 0.0,
        w_config_grad_normal   = 0.0,
        w_config_grad_tangent  = 0.0,
        w_config_grad_perp     = 0.1,
        w_st_estimate_pen      = 0.0,
        w_st_estimate_normal   = 0.0,
        w_st_estimate_tangent  = 0.0,
        w_tangent_jac_d2       = 0.0
    )

    device: str = 'cpu'
    lr: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 0.0
    noise: float = 0.0
    H: int = 128
    k: int = 4  # TODO: rename to vert_n here and in other modules

    epochs: int = 0
    batch: int = 0


def do_train_structured(epochs: int, batch: int, patience: int, scheduler_step_size: int,
                        resume: bool, training = Block2DTraining()):
    if (scheduler_step_size is not None) and (scheduler_step_size > 0):
        training.scheduler_step_size = scheduler_step_size
    training.epochs = epochs
    training.batch = batch

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    tensor_utils.set_default_tensor_type(device)

    bp: Block2DParams = file_utils.load_params(device, 'experiment')

    vertices = torch.zeros(training.k, 2)

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp)
    interaction = StructuredLearnable(system.entities[0], system.entities[1],
                                      vertices, bp.mu, training.net_type, training.H,
                                      training.learn_normal, training.learn_tangent)
    system.resolver.interactions = nn.ModuleList([interaction])

    # Formulate the training loss, as well as reporting losses
    loss = PolyGround2DSurrogate(system, interaction, config=training.surrogate_config)
    prediction_loss = PredictionLoss(system, mode=PredictionMode.XY_ROT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss)

    reporting_losses = [ReportingLoss(loss, sys.maxsize),
                        ReportingLoss(prediction_loss, sys.maxsize),
                        ReportingLoss(trajectory_loss, 15)]
    loss_manager = LossManager(system, [loss], reporting_losses)

    # Specify the optimizer to use
    def optimizer_factory(params):
        return torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)

    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)

    def tensorboard_callback(writer: SummaryWriter, epoch_vars: Dict[str, Number]) -> None:
        tensorboard_callback_structured(writer, epoch_vars, training, bp, system, data_manager)

    # Set up tensorboard logging
    loss_names = ['train_' + name for name in loss.names]
    plot_vars = [loss_names[0], 'valid_surrogate'] + loss_names[1:]
    plot = TensorboardPlot('a_loss', plot_vars, [], True)
    tensorboard = TensorboardManager(callback=tensorboard_callback)
    tensorboard.plots.append(plot)

    hist_data = torch.cat(data_manager.train_runs, dim=1)
    hist_data = hist_data[0, :, 0:3]
    histogram = TensorboardSequentialHistogram(
        'phi', cast(nn.Sequential, interaction.phi_net), hist_data)
    tensorboard.histograms.append(histogram)

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

    trainer.train(epochs=epochs, batch_size=batch, evaluation_loss='valid_pos_int_traj',
                  patience=patience, surpress_printing=True)


@dataclass
class Block2DTrainingE2E:
    device: str = 'cpu'
    lr: float = 1e-3
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    wd: float = 0.0
    noise: float = 0.0

    epochs: int = 0
    batch: int = 0


def do_train_e2e(epochs: int, batch: int, patience: int,
                 scheduler_step_size: int, resume: bool):
    training = Block2DTrainingE2E()
    training.epochs = epochs
    training.batch = batch

    if (scheduler_step_size is not None) and (scheduler_step_size > 0):
        training.scheduler_step_size = scheduler_step_size

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    tensor_utils.set_default_tensor_type(device)

    bp: Block2DParams = file_utils.load_params(device, 'experiment')

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp)
    interaction = DeepLearnable(list(system.entities))
    system.resolver = DirectResolver([interaction])

    # Formulate the training loss, as well as reporting losses
    training_loss = BasicVelocityLoss(system)
    prediction_loss = PredictionLoss(system, mode=PredictionMode.XY_ROT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss)

    reporting_losses = [ReportingLoss(training_loss, sys.maxsize),
                        ReportingLoss(prediction_loss, sys.maxsize),
                        ReportingLoss(trajectory_loss, 15)]
    loss_manager = LossManager(system, [training_loss], reporting_losses)

    # Specify the optimizer to use
    def optimizer_factory(params):
        return torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)

    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)

    def tensorboard_callback(writer: SummaryWriter, epoch_vars: Dict[str, Number]) -> None:
        tensorboard_callback_e2e(writer, epoch_vars, training, bp, system, data_manager)

    # Set up tensorboard logging
    plot = TensorboardPlot('a_loss', ['train_vel_basic', 'valid_vel_basic'], [], True)
    tensorboard = TensorboardManager(callback=tensorboard_callback)
    tensorboard.plots.append(plot)

    hist_data = torch.cat(data_manager.train_runs, dim=1)
    histogram = TensorboardSequentialHistogram(
        'net', cast(nn.Sequential, interaction.interaction), hist_data)
    tensorboard.histograms.append(histogram)

    epoch_prints = [('train_vel_basic', 'train_vel_basic'),
                    ('valid_vel_basic', 'valid_vel_basic'),
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

    trainer.train(epochs=epochs, batch_size=batch, evaluation_loss='valid_pos_int_traj',
                  patience=patience, surpress_printing=False)


@click.command()
@click.option('--epochs', default=100)
@click.option('--batch', default=1)
@click.option('--patience', type=int, default=None)
@click.option('--scheduler_step_size', type=int, default=None)
@click.option('--e2e/--structured', default=False)
@click.option('--resume/--restart', default=False)
def main(epochs: int, batch: int, patience: int,
         scheduler_step_size: int, e2e: bool, resume: bool):
    if e2e:
        do_train_e2e(epochs, batch, patience, scheduler_step_size, resume)
    else:
        do_train_structured(epochs, batch, patience, scheduler_step_size, resume)


if __name__ == "__main__": main()
