import os
import os.path
if __name__ == "__main__":
    # Make pygame renders headless
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import sys

import json

import math
import numpy as np

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import skvideo.io

import time

import subprocess

from argparse import ArgumentParser

from tensorboardX import SummaryWriter
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from contactnets.system import System
from contactnets.train import Trainer, DataManager, TensorboardManager, TensorboardPlot, TensorboardSeriesHistogram, PolyGround2DSurrogate, SurrogateConfig2D, Loss, ReportingLoss, LossManager, PredictionLoss, PredictionMode, TrajectoryLoss, BasicVelocityLoss
from contactnets.interaction import DirectResolver, DirectLearnable, PolyGeometry2D
from contactnets.vis import Visualizer2D

from contactnets.experiments.block2d import sim, Block2DParams, StructuredLearnable, DeepLearnable

from contactnets.utils import file_utils, utils, dirs, system_io

from typing import *
from numbers import Number
from dataclasses import dataclass

import click

import pdb

def save_toss_video(writer: SummaryWriter, system: System, bp: Block2DParams, run: Tensor, name: str) -> None:
    system_io.load_system(run, system)
    sim_results = [system.get_sim_result()]
    system.restart_sim()
    sim_results.append(system.get_sim_result())

    geometry = utils.create_geometry2d(torch.tensor(bp.vertices))

    vis = Visualizer2D(sim_results, [geometry], system.params)
    file_path = vis.render(save_video=True)

    video = skvideo.io.vread(file_path)
    # batch_n = 1 x time_steps x colors x H x W
    # Move color and add empty batch dim
    video = np.expand_dims(np.moveaxis(video, 3, 1), 0)
    # Downsample video
    video = video[:, :, :, 3::4, 3::4]

    writer.add_video(name, video, fps=15)

    os.remove(file_path)

### FIELDS MUST BE DECORATED IN ORDER FOR DATACLASS TO WORK
@dataclass
class Block2DTraining:
    learn_normal: bool = True
    learn_tangent: bool = True

    w_penetration:          float = 0.0
    w_config_grad_normal:   float = 0.0
    w_config_grad_tangent:  float = 0.0
    w_config_grad_perp:     float = 0.1
    w_st_estimate_pen:      float = 0.0
    w_st_estimate_normal:   float = 0.0
    w_st_estimate_tangent:  float = 0.0
    w_tangent_jac_d2:       float = 0.0

    phi_penalties_lin:      bool  = False

    device: str = 'cpu'
    lr: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 0.0
    noise: float = 0.0
    H: int = 128
    k: int = 4

def set_default_tensor_type(device: str):
    if device == 'cpu':
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

def do_train_structured(epochs: int, batch: int, patience: int, scheduler_step_size: int, resume: bool):
    training = Block2DTraining()

    if (scheduler_step_size is not None) and (scheduler_step_size > 0):
        training.scheduler_step_size = scheduler_step_size

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    set_default_tensor_type(training.device)

    bp: Block2DParams = file_utils.load_params(device, 'experiment')

    vertices = torch.zeros(2, training.k)

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp)
    interaction = StructuredLearnable(system.entities[0], system.entities[1],
            vertices, bp.mu, training.H, training.learn_normal, training.learn_tangent)
    system.resolver.interactions = torch.nn.ModuleList([interaction])

    config = SurrogateConfig2D(w_penetration = training.w_penetration,
                               w_config_grad_normal = training.w_config_grad_normal,
                               w_config_grad_tangent = training.w_config_grad_tangent,
                               w_config_grad_perp = training.w_config_grad_perp,
                               w_st_estimate_pen = training.w_st_estimate_pen,
                               w_st_estimate_normal = training.w_st_estimate_normal,
                               w_st_estimate_tangent = training.w_st_estimate_tangent,
                               w_tangent_jac_d2 = training.w_tangent_jac_d2,
                               phi_penalties_lin = training.phi_penalties_lin)

    # Formulate the training loss, as well as reporting losses
    loss = PolyGround2DSurrogate(system, interaction, config=config)
    prediction_loss = PredictionLoss(system, mode=PredictionMode.XY_ROT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss)

    reporting_losses = [ReportingLoss(loss, float('inf')),
                        ReportingLoss(prediction_loss, float('inf')),
                        ReportingLoss(trajectory_loss, 15)]
    loss_manager = LossManager(system, [loss], reporting_losses)

    # Specify the optimizer to use
    optimizer_factory = lambda params: torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)
    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)

    # phi landscape rendering
    def phi_render(writer, normal = True, vrange = None) -> None:
        fig = plt.figure(figsize=(5,5))
        ax = plt.subplot(111)

        points_n = 20

        extent = [-5, 5, -5.0, 5.0, -math.pi, math.pi]

        xs_orig = torch.linspace(extent[0], extent[1], points_n).unsqueeze(1)
        ys_orig = torch.linspace(extent[2], extent[3], points_n).unsqueeze(1)
        ths_orig = torch.linspace(extent[4], extent[5], points_n).unsqueeze(1)

        # Elementwise repeated
        xs = xs_orig.repeat((1, points_n ** 2)).reshape(-1, 1)
        ys = ys_orig.repeat((points_n ** 2, 1))
        ths = ths_orig.repeat((1, points_n)).reshape(-1, 1).repeat((points_n, 1))

        # Vectorwise repeated
        xs_plt = xs_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n,points_n).T
        ths_plt = ths_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n,points_n)

        if normal:
            coords = torch.cat((xs, ys, ths), dim=1)
            phi = interaction.phi_net(coords.unsqueeze(1))
        else:
            coords = torch.cat((ys, xs, ths), dim=1)
            phi = interaction.tangent_net(coords.unsqueeze(1))

        phi = torch.abs(phi.squeeze(1))
        n_cont = phi.shape[1]
        phi2 = phi.reshape((points_n ** 2,points_n, phi.shape[1]))
        best_ys = ys_orig[torch.min(phi2, dim=1).indices].squeeze(2)
        for plt_i in range(n_cont):
            fig = plt.figure(figsize=(10,10))
            #ax = plt.subplot(111)
            ax = fig.gca(projection='3d')
            ys_plt_i = best_ys[:,plt_i].reshape(points_n, points_n).T

            np_cast = lambda t: t.detach().cpu().numpy()
            surf = ax.plot_surface(np_cast(xs_plt), np_cast(ths_plt), np_cast(ys_plt_i), cmap=cm.coolwarm, linewidth=0)
            ax.set_zlim(extent[2], extent[3])
            #ax.zaxis.set_major_locator(LinearLocator(10))
            #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.set_xlabel('x' if normal else 'y')
            ax.set_ylabel('theta')
            ax.set_zlabel('y' if normal else 'x')
            ax.set_title(('phi(' if normal else 'phi_t') + str(plt_i) + ') = 0 surface')
            fig.colorbar(surf, shrink=0.5, aspect=10)
            title_base = 'b_phi/ind_' if normal else 'd_phi_t/ind_'
            writer.add_figure(title_base + str(plt_i), fig)

    def phi_render_slice(writer, vrange = None, normal=True):
        points_n = 100

        extent = [-math.pi, math.pi, -5, 5]

        # xs and ys are not actually x and y config vars, just plot points
        thetas = torch.linspace(extent[0], extent[1], points_n).unsqueeze(1)
        xsys = torch.linspace(extent[2], extent[3], points_n).unsqueeze(1)

        # Elementwise repeated
        thetas = thetas.repeat((1, points_n)).reshape(-1, 1)
        # Vectorwise repeated
        xsys = xsys.repeat((points_n, 1))

        if normal:
            coords = torch.cat((torch.zeros(xsys.shape), xsys, thetas), dim=1).unsqueeze(1)
        else:
            coords = torch.cat((xsys, torch.zeros(xsys.shape), thetas), dim=1).unsqueeze(1)

        phi = interaction.phi_net(coords) if normal else interaction.tangent_net(coords)
        n_cont = phi.shape[2]

        for plt_i in range(n_cont):
            fig = plt.figure(figsize=(6, 5))
            ax = plt.subplot(111)

            phi_i = phi[:, :, plt_i].reshape((points_n, points_n)).t().flip(dims=[0])
            phi_i = phi_i.cpu().detach().numpy()

            phi_range = np.abs(phi_i).max()
            aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
            if vrange == None:
                plt.imshow(phi_i, vmin=-phi_range, vmax=phi_range, cmap=plt.cm.seismic, extent=extent, aspect=aspect)
            else:
                plt.imshow(phi_i, vmin=vrange[0], vmax=vrange[1], cmap=plt.cm.seismic, extent=extent, aspect=aspect)

            plt.xlabel('theta')
            if normal:
                plt.ylabel('y')
                plt.title(f'phi({plt_i}) for x=0 slice')
            else:
                plt.ylabel('x')
                plt.title(f'phi_t({plt_i}) for y=0 slice')

            plt.colorbar()

            title_base = 'c_phi_slice/ind_' if normal else 'e_phi_t_slice/ind_'
            writer.add_figure(title_base + str(plt_i), fig)

    def tensorboard_callback(writer: SummaryWriter, epoch_vars: Dict[str, Number]) -> None:
        if epoch_vars['epoch'] == 0:
            git_file = dirs.out_path('git.txt')
            if os.path.exists(git_file):
                writer.add_text('git', file_utils.read_file(git_file).replace('\n', '  \n'))

            training_text = json.dumps(training.__dict__, sort_keys=True, indent=4).replace('\n', '  \n')
            experiment_text = json.dumps(file_utils.stringify_tensors(bp).__dict__,
                    sort_keys=True, indent=4).replace('\n', '  \n')
            writer.add_text('training', training_text)
            writer.add_text('experiment', experiment_text)

            writer.add_custom_scalars(\
                  {'Prediction': {'Position':   ['Multiline', ['train_pos_pred',        'valid_pos_pred']],
                                  'Angle':      ['Multiline', ['train_angle_pred',      'valid_angle_pred']],
                                  'Velocity':   ['Multiline', ['train_vel_pred',        'valid_vel_pred']],
                                  'Angular Vel':['Multiline', ['train_angle_vel_pred',  'valid_angle_vel_pred']]},
                   'Trajectory': {'Pos Final':  ['Multiline', ['train_pos_final_traj',  'valid_pos_final_traj']],
                                  'Angle Final':['Multiline', ['train_angle_final_traj','valid_angle_final_traj']],
                                  'Pos Int':    ['Multiline', ['train_pos_int_traj',    'valid_pos_int_traj']],
                                  'Angle Int':  ['Multiline', ['train_angle_int_traj',  'valid_angle_int_traj']],
                                  'Pos Bound':  ['Multiline', ['train_pos_bound_traj',  'valid_pos_bound_traj']],
                                  'Angle Bound':['Multiline', ['train_angle_bound_traj','valid_angle_bound_traj']]},
                   'Surrogate': {'Surrogate':   ['Multiline', ['train_surrogate',       'valid_surrogate']]},
                   'Time':      {'Time':        ['Multiline', ['time_train',            'time_log']]}})

        if training.learn_normal:
            phi_render(writer, normal=True)
            phi_render_slice(writer, normal=True)
        if training.learn_tangent:
            phi_render(writer, normal=False)
            phi_render_slice(writer, normal=False)

        for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
            save_toss_video(writer, system, bp, data_manager.train_runs[i], f'f_toss/train{i}')
            save_toss_video(writer, system, bp, data_manager.valid_runs[i], f'f_toss/valid{i}')


    # Set up tensorboard logging
    loss_names = ['train_' + name for name in loss.names]
    plot = TensorboardPlot('a_loss',
            [loss_names[0], 'valid_surrogate'] + loss_names[1:], [], True)
    tensorboard = TensorboardManager(callback=tensorboard_callback)
    tensorboard.plots.append(plot)

    hist_data = torch.cat(data_manager.train_runs, dim=1)
    hist_data = hist_data[:, :, 0:3]
    histogram = TensorboardSeriesHistogram('phi', interaction.phi_net.root.operation, hist_data)
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

    training.epochs = epochs
    training.batch = batch
    file_utils.save_params(training, 'training')

    file_utils.create_empty_directory(dirs.out_path('renders'))
    tensorboard.create_writer(resume=resume)
    tensorboard.launch()

    if resume: trainer.load_training()

    trainer.train(epochs=epochs, batch_size=batch,
            evaluation_loss='valid_pos_int_traj', patience=patience, surpress_printing=True)

@dataclass
class Block2DTrainingE2E:
    device: str = 'cpu'
    lr: float = 1e-3
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    wd: float = 0.0
    noise: float = 0.0

def do_train_e2e(epochs: int, batch: int, patience: int, scheduler_step_size: int, resume: bool):
    training = Block2DTrainingE2E()

    if (scheduler_step_size is not None) and (scheduler_step_size > 0):
        training.scheduler_step_size = scheduler_step_size

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    set_default_tensor_type(training.device)

    bp: Block2DParams = file_utils.load_params(device, 'experiment')

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp)
    interaction = DeepLearnable(system.entities)
    system.resolver = DirectResolver(torch.nn.ModuleList([interaction]))

    # Formulate the training loss, as well as reporting losses
    training_loss = BasicVelocityLoss(system)
    prediction_loss = PredictionLoss(system, mode=PredictionMode.XY_ROT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss)

    reporting_losses = [ReportingLoss(training_loss, float('inf')),
                        ReportingLoss(prediction_loss, float('inf')),
                        ReportingLoss(trajectory_loss, 15)]
    loss_manager = LossManager(system, [training_loss], reporting_losses)

    # Specify the optimizer to use
    optimizer_factory = lambda params: torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)
    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)

    def tensorboard_callback(writer: SummaryWriter, epoch_vars: Dict[str, Number]) -> None:
        if epoch_vars['epoch'] == 0:
            git_file = dirs.out_path('git.txt')
            if os.path.exists(git_file):
                writer.add_text('git', file_utils.read_file(git_file).replace('\n', '  \n'))

            training_text = json.dumps(training.__dict__, sort_keys=True, indent=4).replace('\n', '  \n')
            experiment_text = json.dumps(file_utils.stringify_tensors(bp).__dict__,
                    sort_keys=True, indent=4).replace('\n', '  \n')
            writer.add_text('training', training_text)
            writer.add_text('experiment', experiment_text)

            writer.add_custom_scalars(\
                  {'Prediction': {'Position':   ['Multiline', ['train_pos_pred',        'valid_pos_pred']],
                                  'Angle':      ['Multiline', ['train_angle_pred',      'valid_angle_pred']],
                                  'Velocity':   ['Multiline', ['train_vel_pred',        'valid_vel_pred']],
                                  'Angular Vel':['Multiline', ['train_angle_vel_pred',  'valid_angle_vel_pred']]},
                   'Trajectory': {'Pos Final':  ['Multiline', ['train_pos_final_traj',  'valid_pos_final_traj']],
                                  'Angle Final':['Multiline', ['train_angle_final_traj','valid_angle_final_traj']],
                                  'Pos Int':    ['Multiline', ['train_pos_int_traj',    'valid_pos_int_traj']],
                                  'Angle Int':  ['Multiline', ['train_angle_int_traj',  'valid_angle_int_traj']],
                                  'Pos Bound':  ['Multiline', ['train_pos_bound_traj',  'valid_pos_bound_traj']],
                                  'Angle Bound':['Multiline', ['train_angle_bound_traj','valid_angle_bound_traj']]},
                   'Velocity':  {'Vel Basic':   ['Multiline', ['train_vel_basic',       'valid_vel_basic']]},
                   'Time':      {'Time':        ['Multiline', ['time_train',            'time_log']]}})

        for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
            save_toss_video(writer, system, bp, data_manager.train_runs[i], f'f_toss/train{i}')
            save_toss_video(writer, system, bp, data_manager.valid_runs[i], f'f_toss/valid{i}')

    # Set up tensorboard logging
    plot = TensorboardPlot('a_loss', ['train_vel_basic', 'valid_vel_basic'], [], True)
    tensorboard = TensorboardManager(callback=tensorboard_callback)
    tensorboard.plots.append(plot)

    hist_data = torch.cat(data_manager.train_runs, dim=1)
    histogram = TensorboardSeriesHistogram('net', interaction.interaction_module, hist_data)
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

    training.epochs = epochs
    training.batch = batch
    file_utils.save_params(training, 'training')

    file_utils.create_empty_directory(dirs.out_path('renders'))
    tensorboard.create_writer(resume=resume)
    tensorboard.launch()

    if resume: trainer.load_training()

    trainer.train(epochs=epochs, batch_size=batch,
            evaluation_loss='valid_pos_int_traj', patience=patience, surpress_printing=True)

@click.command()
@click.option('--epochs', default=100)
@click.option('--batch', default=1)
@click.option('--patience', type=int, default=None)
@click.option('--scheduler_step_size', type=int, default=None)
@click.option('--e2e/--surrogate', default=False)
@click.option('--resume/--restart', default=False)
def main(epochs: int, batch: int, patience: int, scheduler_step_size: int, e2e: bool, resume: bool):
    if e2e:
        do_train_e2e(epochs, batch, patience, scheduler_step_size, resume)
    else:
        do_train_structured(epochs, batch, patience, scheduler_step_size, resume)

if __name__ == "__main__":main()
