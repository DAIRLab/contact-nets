import sys
sys.path.append('..')

import os
import os.path

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
from contactnets.train import Trainer, DataManager, TensorboardManager, TensorboardPlot, TensorboardSeriesHistogram, PolyGround3DSurrogate, PolyGround3DConeSurrogate, SurrogateConfig3D, ConeSurrogateConfig3D, Loss, ReportingLoss, LossManager, PredictionLoss, PredictionMode, TrajectoryLoss
from contactnets.interaction import PolyGeometry3D
from contactnets.vis import Visualizer3D

from contactnets.experiments.block3d import sim, Block3DParams, StructuredLearnable

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

    writer.add_video(name, video, fps=15)

    os.remove(file_path)

### FIELDS MUST BE DECORATED IN ORDER FOR DATACLASS TO WORK
@dataclass
class Block3DTraining:
    learn_normal: bool = True
    learn_tangent: bool = True
    
    w_penetration:          float = 0.0
    w_config_grad_normal:   float = 0.0
    w_config_grad_tangent:  float = 0.0
    w_config_grad_perp:     float = 0.1
    w_st_estimate_pen:      float = 0.0
    w_st_estimate_normal:   float = 30.0
    w_st_estimate_tangent:  float = 0.0
    w_tangent_jac_d2:       float = 0.0
    w_contact_threshold:    float = -1

    device: str = 'cpu'
    lr: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 0.01
    noise: float = 0.0
    H: int = 128
    k: int = 8

cone3DTraining = Block3DTraining()
cone3DTraining.learn_normal: bool = True
cone3DTraining.learn_tangent: bool = False

cone3DTraining.w_penetration:          float = 2.0
cone3DTraining.w_config_grad_normal:   float = 0.1
cone3DTraining.w_config_grad_tangent:  float = 0.0
cone3DTraining.w_config_grad_perp:     float = 0.1
cone3DTraining.w_st_estimate_pen:      float = 1.0
cone3DTraining.w_st_estimate_normal:   float = 30.0
cone3DTraining.w_st_estimate_tangent:  float = 0.0
cone3DTraining.w_tangent_jac_d2:       float = 1.0
cone3DTraining.w_contact_threshold:    float = -1

cone3DTraining.device: str = 'cpu'
cone3DTraining.lr: float = 1e-4
cone3DTraining.scheduler_step_size: int = 30
cone3DTraining.scheduler_gamma: float = 0.1
cone3DTraining.wd: float = 1e-2
cone3DTraining.noise: float = 0.0
cone3DTraining.H: int = 128
cone3DTraining.k: int = 8
def do_train(epochs: int, batch: int, patience: int, resume: bool, video: bool = True, socp: bool = False):
    if socp:
        training = cone3DTraining
    else:
        training = Block3DTraining()

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
    if training.k is not 8:
        vertices = torch.zeros(3, training.k)
    else:
        vertices = bp.vertices

    # Modify default system with a learnable polygon-ground interaction
    #pdb.set_trace()
    system = sim.create_empty_system(bp)
    interaction = StructuredLearnable(system.entities[0], system.entities[1],
            vertices, bp.mu, training.H, training.learn_normal, training.learn_tangent, hist_data)
    system.resolver.interactions = torch.nn.ModuleList([interaction])
    if socp:
        config = ConeSurrogateConfig3D(w_penetration = training.w_penetration,
                                       w_config_grad_normal = training.w_config_grad_normal,
                                       w_config_grad_tangent = training.w_config_grad_tangent,
                                       w_config_grad_perp = training.w_config_grad_perp,
                                       w_st_estimate_pen = training.w_st_estimate_pen,
                                       w_st_estimate_normal = training.w_st_estimate_normal,
                                       w_st_estimate_tangent = training.w_st_estimate_tangent,
                                       w_tangent_jac_d2 = training.w_tangent_jac_d2,
                                       w_contact_threshold = training.w_contact_threshold)
    else:
        config = SurrogateConfig3D(w_penetration = training.w_penetration,
                                   w_config_grad_normal = training.w_config_grad_normal,
                                   w_config_grad_tangent = training.w_config_grad_tangent,
                                   w_config_grad_perp = training.w_config_grad_perp,
                                   w_st_estimate_pen = training.w_st_estimate_pen,
                                   w_st_estimate_normal = training.w_st_estimate_normal,
                                   w_st_estimate_tangent = training.w_st_estimate_tangent,
                                   w_tangent_jac_d2 = training.w_tangent_jac_d2,
                                   w_contact_threshold = training.w_contact_threshold)

    # Formulate the training loss, as well as reporting losses
    if socp:
        loss = PolyGround3DConeSurrogate(system, interaction, config=config)
    else:
        loss = PolyGround3DSurrogate(system, interaction, config=config)
    
    prediction_loss = PredictionLoss(system, mode=PredictionMode.XYZ_QUAT) 
    trajectory_loss = TrajectoryLoss(system, prediction_loss)

    #if socp:
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
        xs_plt = xs_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n,points_n).T
        thetas_plt = thetas_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n,points_n)
        
        if normal:
            thetas = torch.cat((thetas, torch.zeros(thetas.shape[0], 2)), dim=1)
            quats = torch.tensor(quat.euler_to_quaternion(thetas, 'yzx'))
            coords = torch.cat((xs, torch.zeros(xs.shape), ys, quats), dim=1)
            phi = interaction.phi_net(coords.unsqueeze(1))
        else:
            thetas = torch.cat((torch.zeros_like(thetas), thetas, torch.zeros_like(thetas)), dim=1)
            quats = torch.tensor(quat.expmap_to_quaternion(thetas.detach().numpy()))
            coords = torch.cat((xs, torch.zeros(xs.shape), ys, quats), dim=1)
            phi = interaction.tangent_net(coords.unsqueeze(1))

            # Get only x coordinates
            phi = phi[:, :, 0::2]

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
            ax.set_title(('phi(' if normal else 'phi_t') + str(plt_i) + ') = 0 surface (y=0)')
            fig.colorbar(surf, shrink=0.5, aspect=10)
            title_base = 'b_phi/ind_' if normal else 'd_phi_t/ind_' 
            writer.add_figure(title_base + str(plt_i), fig)

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

        if video:
            for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
                save_toss_video(writer, system, bp, data_manager.train_runs[i], f'f_toss/train{i}')
                save_toss_video(writer, system, bp, data_manager.valid_runs[i], f'f_toss/valid{i}')
    

    # Set up tensorboard logging
    loss_names = ['train_' + name for name in loss.names]
    plot = TensorboardPlot('a_loss',
            [loss_names[0], 'valid_surrogate'] + loss_names[1:], [], True)
    tensorboard = TensorboardManager(callback=tensorboard_callback)
    tensorboard.plots.append(plot)

    
    histogram = TensorboardSeriesHistogram('phi', interaction.phi_net.root.operation, hist_data)
    histogram2 = TensorboardSeriesHistogram('phi_t', interaction.tangent_net.root.operation, hist_data) 
    tensorboard.histograms.append(histogram)
    tensorboard.histograms.append(histogram2)

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

    trainer.train(epochs=epochs, batch_size=batch, evaluation_loss='valid_pos_int_traj',
            patience=patience, surpress_printing=False)

@click.command()
@click.option('--epochs', default=100)
@click.option('--batch', default=1)
@click.option('--patience', type=int, default=None)
@click.option('--resume/--restart', default=False)
@click.option('--resume/--restart', default=False)
@click.option('--video/--no-video', default=True)
@click.option('--socp/--qp', default=False)
def main(epochs: int, batch: int, patience: int, resume: bool, video: bool, socp: bool):
    do_train(epochs, batch, patience, resume, video, socp)

if __name__ == "__main__":main()
