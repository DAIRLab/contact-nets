import sys
sys.path.append('..')

import os
import os.path
import contextlib

import json

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
from contactnets.train import Trainer, DataManager, TensorboardManager, TensorboardPlot, TensorboardSeriesHistogram, PolyGround3DSurrogate, SurrogateConfig3D, Loss, ReportingLoss, LossManager, PredictionLoss, PredictionMode, TrajectoryLoss, BasicVelocityLoss
from contactnets.interaction import PolyGeometry3D, PolyGround3D, DirectResolver, DirectLearnable
from contactnets.vis import Visualizer3D

from contactnets.experiments.block3d import sim, Block3DParams, StructuredLearnable, DeepLearnable

from contactnets.utils import file_utils, utils, dirs, system_io
from contactnets.utils import quaternion as quat

from typing import *
from numbers import Number
from dataclasses import dataclass

import click

import pdb

def save_toss_video(writer: SummaryWriter, system: System, bp: Block3DParams, run: Tensor, name: str) -> None:
    system_io.load_system(run, system)
    sim_results = [system.get_sim_result()]
    system.restart_sim()
    sim_results.append(system.get_sim_result())

    geometry = PolyGeometry3D(torch.tensor(bp.vertices))

    vis = Visualizer3D(sim_results, [geometry], system.params)
    file_path = vis.render(save_video=True, headless=True)

    video = skvideo.io.vread(file_path)

    # batch_n = 1 x time_steps x colors x H x W
    # Move color and add empty batch dim
    video = np.expand_dims(np.moveaxis(video, 3, 1), 0)
    # Downsample video
    video = video[:, :, :, 3::4, 3::4]

    # Surpress prints
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        writer.add_video(name, video, fps=15)
    
    os.remove(file_path)

### FIELDS MUST BE DECORATED IN ORDER FOR DATACLASS TO WORK
@dataclass
class Block3DTraining:
    learn_normal: bool = True
    learn_tangent: bool = True

    polytope: bool = True

    # TODO: implement with surrogate config member variable
    w_comp_n:               float = 1.0
    w_comp_t:               float = 1.0
    w_match:                float = 1.0
    w_cone:                 float = 1.0
    w_penetration_slack:    float = 1.0

    w_penetration:          float = 0.0
    w_config_grad_normal:   float = 0.3
    w_config_grad_tangent:  float = 0.0
    w_config_grad_perp:     float = 0.3
    w_st_estimate_pen:      float = 0.0
    w_st_estimate_normal:   float = 0.0
    w_st_estimate_tangent:  float = 0.0
    w_tangent_jac_d2:       float = 0.0
    w_contact_threshold:    float = 0.5
    # w_contact_threshold:    float = -1.0

    elastic:                bool  = True

    device: str = 'cpu'
    lr: float = 5e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 0.0
    noise: float = 0.0
    H: int = 256
    k: int = 8

def do_train(epochs: int, batch: int, patience: int, resume: bool, training = Block3DTraining(), video: bool = True):
    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    if training.device == 'cpu':
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)
    hist_data = torch.cat(data_manager.train_runs, dim=1)
    hist_data = hist_data[:, :, 0:7]

    bp: Block3DParams = file_utils.load_params(device, 'experiment')
    bp.restitution = torch.tensor(0.0)
    if training.k != 8:
        vertices = torch.zeros(3, training.k)
    else:
        vertices = bp.vertices

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp, elastic=training.elastic)
    interaction = StructuredLearnable(system.entities[0], system.entities[1], training.polytope,
            vertices, bp.mu, training.H, training.learn_normal, training.learn_tangent, hist_data)
    ground_truth_phi = PolyGround3D(system.entities[0], system.entities[1],
            vertices, bp.mu).compute_phi
    system.resolver.interactions = torch.nn.ModuleList([interaction])


    config = SurrogateConfig3D(w_comp_n = training.w_comp_n,
                               w_comp_t = training.w_comp_t,
                               w_match = training.w_match,
                               w_cone = training.w_cone,
                               w_penetration_slack = training.w_penetration_slack,
                               w_penetration = training.w_penetration,
                               w_config_grad_normal = training.w_config_grad_normal,
                               w_config_grad_tangent = training.w_config_grad_tangent,
                               w_config_grad_perp = training.w_config_grad_perp,
                               w_st_estimate_pen = training.w_st_estimate_pen,
                               w_st_estimate_normal = training.w_st_estimate_normal,
                               w_st_estimate_tangent = training.w_st_estimate_tangent,
                               w_tangent_jac_d2 = training.w_tangent_jac_d2,
                               w_contact_threshold = training.w_contact_threshold)

    # Formulate the training loss, as well as reporting losses
    loss = PolyGround3DSurrogate(system, interaction, config=config)

    prediction_loss = PredictionLoss(system, mode=PredictionMode.XYZ_QUAT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss, ground_truth_phi=ground_truth_phi)

    reporting_losses = [ReportingLoss(loss, 1000),
                        ReportingLoss(prediction_loss, 1000),
                        ReportingLoss(trajectory_loss, 15)]

    loss_manager = LossManager(system, [loss], reporting_losses)

    # Specify the optimizer to use
    optimizer_factory = lambda params: torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)
    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

    # phi landscape rendering
    def phi_render(writer, normal = True, vrange = None) -> None:
        fig = plt.figure(figsize=(5,5))
        ax = plt.subplot(111)

        points_n = 20

        if normal:
            #extent = [-5, 5, 0.8, 1.6, -math.pi, math.pi]
            extent = [-5, 5, -5.0, 5.0, -math.pi, math.pi]
        else:
            extent = [-5, 5, -5., 5., -math.pi, math.pi]

        xs_orig = torch.linspace(extent[0], extent[1], points_n).unsqueeze(1)
        ys_orig = torch.linspace(extent[2], extent[3], points_n).unsqueeze(1) # Actually zs
        thetas_orig = torch.linspace(extent[4], extent[5], points_n).unsqueeze(1)

        # Elementwise repeated
        xs = xs_orig.repeat((1, points_n ** 2)).reshape(-1, 1)
        ys = ys_orig.repeat((points_n ** 2, 1))
        thetas = thetas_orig.repeat((1, points_n)).reshape(-1, 1).repeat((points_n, 1))

        # Vectorwise repeated
        thetas_plt = thetas_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n,points_n)

        if normal:
            thetas = torch.cat((torch.zeros_like(thetas), thetas, torch.zeros_like(thetas)), dim=1)
            quats = torch.tensor(quat.expmap_to_quaternion(thetas.detach().numpy()))
            coords = torch.cat((xs, torch.zeros(xs.shape), ys, quats), dim=1)

            phi = interaction.phi_net(coords.unsqueeze(1))

            xs_plt = xs_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n,points_n).T
        else:
            thetas = torch.cat((torch.zeros_like(thetas), thetas, torch.zeros_like(thetas)), dim=1)
            quats = torch.tensor(quat.expmap_to_quaternion(thetas.detach().numpy()))
            coords = torch.cat((ys, torch.zeros(ys.shape), xs, quats), dim=1)

            phi = interaction.tangent_net(coords.unsqueeze(1))

            # Get only x coordinates
            phi = phi[:, :, 0::2]

            xs_plt = xs_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n,points_n).T


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
            surf = ax.plot_surface(np_cast(xs_plt), np_cast(thetas_plt), np_cast(ys_plt_i), cmap=cm.coolwarm, linewidth=0)
            ax.set_zlim(extent[2], extent[3])
            ax.set_xlabel('x' if normal else 'z')
            ax.set_ylabel('theta')
            ax.set_zlabel('z' if normal else 'x')
            ax.set_title(('phi(' if normal else 'phi_t_x') + str(plt_i) + ') = 0 surface (y=0)')
            fig.colorbar(surf, shrink=0.5, aspect=10)
            title_base = 'b_phi/ind_' if normal else 'd_phi_t_x/ind_'
            writer.add_figure(title_base + str(plt_i), fig)
            plt.close(fig)

    def phi_render_slice(writer, vrange = None, normal=True):
        points_n = 100

        if normal:
            #extent = [-math.pi, math.pi, 0.8, 1.6]
            extent = [-math.pi, math.pi, -5, 5]
        else:
            extent = [-math.pi, math.pi, -5, 5]

        # xs and ys are not actually x and y config vars, just plot points
        thetas = torch.linspace(extent[0], extent[1], points_n).unsqueeze(1)
        xsys = torch.linspace(extent[2], extent[3], points_n).unsqueeze(1)

        # Elementwise repeated
        thetas = thetas.repeat((1, points_n)).reshape(-1, 1)
        # Vectorwise repeated
        xsys = xsys.repeat((points_n, 1))

        if normal:
            thetas = torch.cat((torch.zeros_like(thetas), thetas, torch.zeros_like(thetas)), dim=1)
            quats = torch.tensor(quat.expmap_to_quaternion(thetas.detach().numpy()))
            coords = torch.cat((torch.zeros(xsys.shape), torch.zeros(xsys.shape), xsys, quats), dim=1)

            phi = interaction.phi_net(coords.unsqueeze(1))
        else:
            thetas = torch.cat((torch.zeros_like(thetas), thetas, torch.zeros_like(thetas)), dim=1)
            quats = torch.tensor(quat.expmap_to_quaternion(thetas.detach().numpy()))
            coords = torch.cat((xsys, torch.zeros(xsys.shape), torch.zeros(xsys.shape), quats), dim=1)

            phi = interaction.tangent_net(coords.unsqueeze(1))

            # Get only x coordinates
            phi = phi[:, :, 0::2]

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
                plt.ylabel('z')
                plt.title(f'phi({plt_i}) for x=0,y=0 slice')
            else:
                plt.ylabel('x')
                plt.title(f'phi_t_x({plt_i}) for y=0 slice')

            plt.colorbar()

            title_base = 'c_phi_slice/ind_' if normal else 'e_phi_t_slice/ind_'
            writer.add_figure(title_base + str(plt_i), fig)

            plt.close(fig)

    def log_corner_figure(writer, corners, axis, title):
        fig = plt.figure(figsize=(10,10))

        ax = plt.subplot(111)

        plt.grid()

        corners = corners.cpu().detach().numpy()

        cm = plt.cm.get_cmap('RdYlBu')
        sc = ax.scatter(corners[(0 + axis) % 3, :],
                        corners[(1 + axis) % 3, :],
                        c=corners[(2 + axis) % 3, :],
                        s=200, cmap=cm)
        cbar = plt.colorbar(sc)

        labels = ['x', 'y', 'z']

        ax.set_xlabel(labels[(0 + axis) % 3])
        ax.set_ylabel(labels[(1 + axis) % 3])
        cbar.ax.set_ylabel(labels[(2 + axis) % 3])

        ax.set(xlim=(-3.0, 3.0), ylim=(-3.0, 3.0))

        writer.add_figure(title, fig)

        plt.close(fig)

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
                  {'Trajectory': {'Pos Int (percent)':    ['Multiline', ['train_pos_int_traj',        'valid_pos_int_traj']],
                                  'Angle Int (degree)':   ['Multiline', ['train_angle_int_traj',      'valid_angle_int_traj']],
                                  'Pen Int (percent)':    ['Multiline', ['train_penetration_int_traj','valid_penetration_int_traj']]},
                   'Surrogate': {'Surrogate':             ['Multiline', ['train_surrogate',           'valid_surrogate']]},
                   'Time':      {'Time':                  ['Multiline', ['time_train',                'time_log']]}})
        
        if training.learn_normal:
            phi_render(writer, normal=True)
            phi_render_slice(writer, normal=True)

        if training.learn_tangent:
            phi_render(writer, normal=False)
            phi_render_slice(writer, normal=False)

        if video:
            for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
                save_toss_video(writer, system, bp, data_manager.train_runs[i], f'f_toss(red=data,blue=prediciton)/train{i}')
                save_toss_video(writer, system, bp, data_manager.valid_runs[i], f'f_toss(red=data,blue=prediciton)/valid{i}')
        
        if training.polytope:
            normal_corners = interaction.phi_net.root.operation.jac_modules[0].operation.vertices
            tangent_corners = interaction.tangent_net.root.operation.jac_modules[0].operation.vertices
        else:
            # normal_corners = interaction.phi_net.root.operation.jac_modules[0].operation.vertex_net.operation.bias.reshape(8, 3).t()
            # tangent_corners = interaction.tangent_net.root.operation.jac_modules[0].operation.vertex_net.operation.bias.reshape(8, 3).t()
            normal_corners = interaction.phi_net.root.operation.jac_modules[0].operation.jac_modules[0].operation.vertices
            tangent_corners = interaction.tangent_net.root.operation.jac_modules[0].operation.jac_modules[0].operation.vertices

        log_corner_figure(writer, normal_corners, 0, 'g_normal_x')
        log_corner_figure(writer, normal_corners, 1, 'g_normal_y')
        log_corner_figure(writer, normal_corners, 2, 'g_normal_z')

        log_corner_figure(writer, tangent_corners, 0, 'h_tangent_x')
        log_corner_figure(writer, tangent_corners, 1, 'h_tangent_y')
        log_corner_figure(writer, tangent_corners, 2, 'h_tangent_z')

    # Set up tensorboard logging
    loss_names = ['train_' + name for name in loss.names]
    plot = TensorboardPlot('a_loss',
            [loss_names[0], 'valid_surrogate'] + loss_names[1:], [], True)
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

    training.epochs = epochs
    training.batch = batch
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

def do_train_e2e(epochs: int, batch: int, patience: int, resume: bool, video: bool = True):
    training = Block3DTrainingE2E()

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    if training.device == 'cpu':
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)
    hist_data = torch.cat(data_manager.train_runs, dim=1)

    bp: Block3DParams = file_utils.load_params(device, 'experiment')

    # Modify default system with a learnable polygon-ground interaction
    system = sim.create_empty_system(bp)
    interaction = DeepLearnable(system.entities, H=training.H,
                                depth=training.depth, data=hist_data)
    ground_truth_phi = PolyGround3D(system.entities[0], system.entities[1],
            bp.vertices, bp.mu).compute_phi
    system.resolver = DirectResolver(torch.nn.ModuleList([interaction]))

    # Formulate the training loss, as well as reporting losses
    training_loss = BasicVelocityLoss(system)
    prediction_loss = PredictionLoss(system, mode=PredictionMode.XYZ_QUAT)
    trajectory_loss = TrajectoryLoss(system, prediction_loss, ground_truth_phi=ground_truth_phi)

    reporting_losses = [ReportingLoss(training_loss, 1000),
                        ReportingLoss(prediction_loss, 1000),
                        ReportingLoss(trajectory_loss, 15)]
    loss_manager = LossManager(system, [training_loss], reporting_losses)

    # Specify the optimizer to use
    optimizer_factory = lambda params: torch.optim.AdamW(params, lr=training.lr, weight_decay=training.wd)
    def scheduler_factory(optimizer):
        return StepLR(optimizer, gamma=training.scheduler_gamma,
                      step_size=training.scheduler_step_size)

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
                  {'Trajectory': {'Pos Int (percent)':    ['Multiline', ['train_pos_int_traj',        'valid_pos_int_traj']],
                                  'Angle Int (degree)':   ['Multiline', ['train_angle_int_traj',      'valid_angle_int_traj']],
                                  'Pen Int (percent)':    ['Multiline', ['train_penetration_int_traj','valid_penetration_int_traj']]},
                   'Loss':      {'Loss':                  ['Multiline', ['train_vel_basic',           'valid_vel_basic']]},
                   'Time':      {'Time':                  ['Multiline', ['time_train',                'time_log']]}})

        if video:
            for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
                save_toss_video(writer, system, bp, data_manager.train_runs[i], f'f_toss(red=data,blue=prediciton)/train{i}')
                save_toss_video(writer, system, bp, data_manager.valid_runs[i], f'f_toss(red=data,blue=prediciton)/valid{i}')

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

    training.epochs = epochs
    training.batch = batch
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
        do_train_e2e(epochs, batch, patience, resume, video)
    else:
        do_train(epochs, batch, patience, resume, video)

if __name__ == "__main__":main()
