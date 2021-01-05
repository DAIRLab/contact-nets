from dataclasses import dataclass
from numbers import Number
import pdb  # noqa
from typing import Dict

import click
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from contactnets.experiments.block3d import (Block3DParams, DeepLearnable, StructuredLearnable,
                                             sim)
from contactnets.experiments.block3d.train_utils import (tensorboard_callback_e2e,
                                                         tensorboard_callback_structured)
from contactnets.interaction import DirectResolver, PolyGround3D
from contactnets.train import (BasicVelocityLoss, DataManager, LossManager,
                               PolyGround3DSurrogate, PredictionLoss, PredictionMode,
                               ReportingLoss, SurrogateConfig3D, TensorboardManager,
                               TensorboardPlot, Trainer, TrajectoryLoss)
from contactnets.utils import dirs, file_utils, tensor_utils


# FIELDS MUST BE DECORATED IN ORDER FOR DATACLASS TO WORK
@dataclass
class Block3DTraining:
    learn_normal: bool = True
    learn_tangent: bool = True

    # 'poly' or 'deepvertex' or 'deep'
    net_type: str = 'poly'

    surrogate_config: SurrogateConfig3D = SurrogateConfig3D(
        w_comp_n               = 1.0,
        w_comp_t               = 1.0,
        w_match                = 0.1,
        w_cone                 = 1.0,
        w_penetration_slack    = 1.0,

        w_penetration          = 0.0,
        w_config_grad_normal   = 0.3,
        w_config_grad_tangent  = 0.0,
        w_config_grad_perp     = 0.3,
        w_st_estimate_pen      = 0.0,
        w_st_estimate_normal   = 0.0,
        w_st_estimate_tangent  = 0.0,
        w_tangent_jac_d2       = 0.0,
        w_contact_threshold    = -0.0,

        robust_sqrt            = True
    )

    elastic: float = False
    device: str = 'cpu'
    lr: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 1e-2
    noise: float = 0.0
    H: int = 256
    k: int = 8

    epochs: int = 0
    batch: int = 0


def do_train_structured(epochs: int, batch: int, patience: int, resume: bool,
                        training = Block3DTraining(), video: bool = True):
    training.epochs = epochs
    training.batch = batch

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    tensor_utils.set_default_tensor_type(device)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)
    hist_data = torch.cat(data_manager.train_runs, dim=1)
    hist_data = hist_data[0, :, 0:7]

    bp: Block3DParams = file_utils.load_params(device, 'experiment')
    bp.restitution = torch.tensor(0.0)
    vertices = bp.vertices if training.k == 8 else torch.zeros(training.k, 3)

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp, elastic=training.elastic)
    interaction = StructuredLearnable(system.entities[0], system.entities[1], vertices, bp.mu,
                                      training.net_type, training.H,
                                      training.learn_normal, training.learn_tangent, hist_data)
    ground_truth_phi = PolyGround3D(system.entities[0], system.entities[1],
                                    vertices, bp.mu).compute_phi
    system.resolver.interactions = nn.ModuleList([interaction])

    # Formulate the training loss, as well as reporting losses
    loss = PolyGround3DSurrogate(system, interaction, config=training.surrogate_config)

    prediction_loss = PredictionLoss(system, mode=PredictionMode.XYZ_QUAT)
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
        tensorboard_callback_structured(writer, epoch_vars, training,
                                        bp, system, data_manager, video)

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


@dataclass
class Block3DTrainingE2E:
    device: str = 'cpu'
    lr: float = 3e-5
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 0.001
    noise: float = 0.0
    H: int = 256
    depth: int = 4

    epochs: int = 0
    batch: int = 0


def do_train_e2e(epochs: int, batch: int, patience: int, resume: bool, video: bool = True):
    training = Block3DTrainingE2E()
    training.epochs = epochs
    training.batch = batch

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    tensor_utils.set_default_tensor_type(device)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)
    hist_data = torch.cat(data_manager.train_runs, dim=1)

    bp: Block3DParams = file_utils.load_params(device, 'experiment')

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp)

    # decide whether to do MDN or not
    interaction = DeepLearnable(list(system.entities), H=training.H,
                                depth=training.depth, data=hist_data)
    ground_truth_phi = PolyGround3D(system.entities[0], system.entities[1],
                                    bp.vertices, bp.mu).compute_phi
    system.resolver = DirectResolver([interaction])

    # Formulate the training loss, as well as reporting losses
    training_loss = BasicVelocityLoss(system)
    prediction_loss = PredictionLoss(system, mode=PredictionMode.XYZ_QUAT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss, ground_truth_phi=ground_truth_phi)

    reporting_losses = [ReportingLoss(training_loss, 1000),
                        ReportingLoss(prediction_loss, 1000),
                        ReportingLoss(trajectory_loss, 15)]
    loss_manager = LossManager(system, [training_loss], reporting_losses)

    # Specify the optimizer to use
    def optimizer_factory(params):
        return torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)

    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

    def tensorboard_callback(writer: SummaryWriter, epoch_vars: Dict[str, Number]) -> None:
        tensorboard_callback_e2e(writer, epoch_vars, training, bp, system, data_manager, video)

    # Set up tensorboard logging
    plot = TensorboardPlot('a_loss', ['train_vel_basic', 'valid_vel_basic'], [], True)
    tensorboard = TensorboardManager(callback=tensorboard_callback)
    tensorboard.plots.append(plot)

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

    trainer.train(epochs=epochs, batch_size=batch, evaluation_loss='valid_vel_basic',
                  patience=patience, surpress_printing=False)


@click.command()
@click.option('--epochs', default=100)
@click.option('--batch', default=1)
@click.option('--patience', type=int, default=None)
@click.option('--resume/--restart', default=False)
@click.option('--resume/--restart', default=False)
@click.option('--video/--no-video', default=True)
@click.option('--e2e/--structured', default=False)
def main(epochs: int, batch: int, patience: int, resume: bool,
         video: bool, e2e: bool):
    if e2e:
        do_train_e2e(epochs, batch, patience, resume, video=video)
    else:
        do_train_structured(epochs, batch, patience, resume, video=video)


if __name__ == "__main__": main()
