import contextlib
import copy
import json
import math
from numbers import Number
import os
import os.path
import pdb  # noqa
from typing import TYPE_CHECKING, Dict, Tuple, cast

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
import skvideo.io
from tensorboardX import SummaryWriter
import torch
from torch import Tensor
import torch.nn as nn

from contactnets.experiments.block3d import Block3DParams
from contactnets.experiments.block3d.structured_learnable import StructuredLearnable
from contactnets.interaction import PolyGeometry3D
from contactnets.system import System
from contactnets.train import DataManager
from contactnets.utils import dirs, file_utils
from contactnets.utils import quaternion as quat
from contactnets.utils import system_io
from contactnets.vis import Visualizer3D

if TYPE_CHECKING:
    from contactnets.experiments.block3d.train import Block3DTraining, Block3DTrainingE2E


def save_toss_video(writer: SummaryWriter, system: System, bp: Block3DParams,
                    run: Tensor, name: str) -> None:
    system_io.load_system(run, system)
    sim_results = [system.get_sim_result()]
    system.restart_sim()
    sim_results.append(system.get_sim_result())

    geometry = PolyGeometry3D(bp.vertices)

    vis = Visualizer3D(sim_results, [geometry], system.params)
    file_path = vis.render(save_video=True, headless=True)

    if file_path is not None:
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


def phi_render(interaction: StructuredLearnable, writer: SummaryWriter,
               normal = True, vrange: Tuple[float, float] = None) -> None:

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)

    points_n = 20

    extent = [-5, 5, -5., 5., -math.pi, math.pi]

    xs_orig = torch.linspace(extent[0], extent[1], points_n).unsqueeze(1)
    ys_orig = torch.linspace(extent[2], extent[3], points_n).unsqueeze(1)  # Actually zs
    thetas_orig = torch.linspace(extent[4], extent[5], points_n).unsqueeze(1)

    # Elementwise repeated
    xs = xs_orig.repeat((1, points_n ** 2)).reshape(-1, 1)
    ys = ys_orig.repeat((points_n ** 2, 1))
    thetas = thetas_orig.repeat((1, points_n)).reshape(-1, 1).repeat((points_n, 1))

    # Vectorwise repeated
    thetas_plt = thetas_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n, points_n)

    thetas = torch.cat((torch.zeros_like(thetas), thetas, torch.zeros_like(thetas)), dim=1)
    quats = torch.tensor(quat.expmap_to_quaternion(thetas.detach().numpy()))
    if normal:
        coords = torch.cat((xs, torch.zeros(xs.shape), ys, quats), dim=1)
        phi = interaction.phi_net(coords)
        xs_plt = xs_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n, points_n).T
    else:
        coords = torch.cat((ys, torch.zeros(ys.shape), xs, quats), dim=1)
        phi = interaction.tangent_net(coords)
        phi = phi[:, 0::2]  # Get only x coordinates
        xs_plt = xs_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n, points_n).T

    phi = torch.abs(phi)
    n_cont = phi.shape[1]
    phi2 = phi.reshape((points_n ** 2, points_n, phi.shape[1]))
    best_ys = ys_orig[torch.min(phi2, dim=1).indices].squeeze(2)
    for plt_i in range(n_cont):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ys_plt_i = best_ys[:, plt_i].reshape(points_n, points_n).T

        def np_cast(t): return t.detach().cpu().numpy()
        surf = ax.plot_surface(np_cast(xs_plt), np_cast(thetas_plt),
                               np_cast(ys_plt_i), cmap=cm.coolwarm, linewidth=0)
        ax.set_zlim(extent[2], extent[3])
        ax.set_xlabel('x' if normal else 'z')
        ax.set_ylabel('theta')
        ax.set_zlabel('z' if normal else 'x')
        ax.set_title(('phi(' if normal else 'phi_t_x') + str(plt_i) + ') = 0 surface (y=0)')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        title_base = 'b_phi/ind_' if normal else 'd_phi_t_x/ind_'
        writer.add_figure(title_base + str(plt_i), fig)
        plt.close(fig)


def phi_render_slice(interaction: StructuredLearnable, writer: SummaryWriter,
                     vrange: Tuple[float, float] = None, normal=True):
    points_n = 100

    extent = [-math.pi, math.pi, -5, 5]

    # xs and ys are not actually x and y config vars, just plot points
    thetas = torch.linspace(extent[0], extent[1], points_n).unsqueeze(1)
    xsys = torch.linspace(extent[2], extent[3], points_n).unsqueeze(1)

    # Elementwise repeated
    thetas = thetas.repeat((1, points_n)).reshape(-1, 1)
    # Vectorwise repeated
    xsys = xsys.repeat((points_n, 1))

    thetas = torch.cat((torch.zeros_like(thetas), thetas, torch.zeros_like(thetas)), dim=1)
    quats = torch.tensor(quat.expmap_to_quaternion(thetas.detach().numpy()))
    if normal:
        coords = torch.cat((torch.zeros(xsys.shape), torch.zeros(xsys.shape),
                            xsys, quats), dim=1)
        phi = interaction.phi_net(coords)
    else:
        coords = torch.cat((xsys, torch.zeros(xsys.shape), torch.zeros(xsys.shape),
                            quats), dim=1)
        phi = interaction.tangent_net(coords)
        phi = phi[:, 0::2]  # Get only x coordinates

    n_cont = phi.shape[1]

    for plt_i in range(n_cont):
        fig = plt.figure(figsize=(6, 5))
        plt.subplot(111)

        phi_i = phi[:, plt_i].reshape((points_n, points_n)).t().flip(dims=[0])
        phi_i = phi_i.cpu().detach().numpy()

        phi_range = np.abs(phi_i).max()
        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        if vrange is None: vrange = (-phi_range, phi_range)
        plt.imshow(phi_i, vmin=vrange[0], vmax=vrange[1],
                   cmap=plt.cm.seismic, extent=extent, aspect=aspect)

        plt.xlabel('theta')
        plt.ylabel('z' if normal else 'x')
        plt.title(f'phi({plt_i}) for x=0,y=0' if normal else f'phi_t_x({plt_i}) for y=0')

        plt.colorbar()

        title_base = 'c_phi_slice/ind_' if normal else 'e_phi_t_slice/ind_'
        writer.add_figure(title_base + str(plt_i), fig)

        plt.close(fig)


def log_corner_figure(writer: SummaryWriter, corners: Tensor, axis: int, title: str):
    fig = plt.figure(figsize=(10, 10))

    ax = plt.subplot(111)

    plt.grid()

    corners = corners.cpu().detach().numpy()

    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax.scatter(corners[:, (0 + axis) % 3],
                    corners[:, (1 + axis) % 3],
                    c=corners[:, (2 + axis) % 3],
                    s=200, cmap=cm)
    cbar = plt.colorbar(sc)

    labels = ['x', 'y', 'z']

    ax.set_xlabel(labels[(0 + axis) % 3])
    ax.set_ylabel(labels[(1 + axis) % 3])
    cbar.ax.set_ylabel(labels[(2 + axis) % 3])

    ax.set(xlim=(-3.0, 3.0), ylim=(-3.0, 3.0))

    writer.add_figure(title, fig)

    plt.close(fig)


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


def tensorboard_callback_structured(writer: SummaryWriter, epoch_vars: Dict[str, Number],
                                    training: 'Block3DTraining', bp: Block3DParams,
                                    system: System, data_manager: DataManager,
                                    video: bool) -> None:
    interaction: StructuredLearnable = system.resolver.interactions[0]

    if epoch_vars['epoch'] == 0:
        git_file = dirs.out_path('git.txt')
        if os.path.exists(git_file):
            writer.add_text('git', file_utils.read_file(git_file).replace('\n', '  \n'))

        training_text = json.dumps(training.__dict__, sort_keys=True, indent=4,
                                   default=lambda x: x.__dict__).replace('\n', '  \n')
        experiment_text = json.dumps(file_utils.listify_tensors(copy.copy(bp)).__dict__,
                                     sort_keys=True, indent=4).replace('\n', '  \n')
        writer.add_text('training', training_text)
        writer.add_text('experiment', experiment_text)

        writer.add_custom_scalars(structured_custom_scalars)

    if training.learn_normal:
        phi_render(interaction, writer, normal=True)
        phi_render_slice(interaction, writer, normal=True)

    if training.learn_tangent:
        phi_render(interaction, writer, normal=False)
        phi_render_slice(interaction, writer, normal=False)

    if video:
        for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
            save_toss_video(writer, system, bp, data_manager.train_runs[i],
                            f'f_toss(red=data,blue=prediciton)/train{i}')
            save_toss_video(writer, system, bp, data_manager.valid_runs[i],
                            f'f_toss(red=data,blue=prediciton)/valid{i}')

    if training.net_type == 'poly':
        normal_corners = interaction.phi_net.points
        tangent_corners = interaction.tangent_net.points
    elif training.net_type == 'deepvertex':
        phi_net = cast(nn.Sequential, interaction.phi_net)
        tangent_net = cast(nn.Sequential, interaction.tangent_net)
        normal_corners = phi_net[0].module.bias.reshape(8, 3).t()
        tangent_corners = tangent_net[0].module.bias.reshape(8, 3).t()
    elif training.net_type == 'deep':
        phi_net_sequential = cast(nn.Sequential, interaction.phi_net.module_list)
        tangent_net_sequential = cast(nn.Sequential, interaction.tangent_net.module_list)
        normal_corners = phi_net_sequential[1].points
        tangent_corners = tangent_net_sequential[1].points
    else:
        raise Exception('Network type not recognized')

    normal = cast(Tensor, normal_corners)
    tangent = cast(Tensor, tangent_corners)

    log_corner_figure(writer, normal, 0, 'g_normal_x')
    log_corner_figure(writer, normal, 1, 'g_normal_y')
    log_corner_figure(writer, normal, 2, 'g_normal_z')

    log_corner_figure(writer, tangent, 0, 'h_tangent_x')
    log_corner_figure(writer, tangent, 1, 'h_tangent_y')
    log_corner_figure(writer, tangent, 2, 'h_tangent_z')


e2e_custom_scalars = {
    'Trajectory': {
        'Pos Int (percent)':  [ct, ['train_pos_int_traj',         'valid_pos_int_traj']],
        'Angle Int (degree)': [ct, ['train_angle_int_traj',       'valid_angle_int_traj']],
        'Pen Int (percent)':  [ct, ['train_penetration_int_traj', 'valid_penetration_int_traj']]
    },
    'Loss':      {'Loss':    [ct, ['train_vel_basic',             'valid_vel_basic']]},
    'Time':      {'Time':    [ct, ['time_train',                  'time_log']]}}


def tensorboard_callback_e2e(writer: SummaryWriter, epoch_vars: Dict[str, Number],
                             training: 'Block3DTrainingE2E', bp: Block3DParams,
                             system: System, data_manager: DataManager, video: bool) -> None:
    if epoch_vars['epoch'] == 0:
        git_file = dirs.out_path('git.txt')
        if os.path.exists(git_file):
            writer.add_text('git', file_utils.read_file(git_file).replace('\n', '  \n'))

        training_text = json.dumps(training.__dict__, sort_keys=True,
                                   indent=4).replace('\n', '  \n')
        experiment_text = json.dumps(file_utils.listify_tensors(copy.copy(bp)).__dict__,
                                     sort_keys=True, indent=4).replace('\n', '  \n')
        writer.add_text('training', training_text)
        writer.add_text('experiment', experiment_text)

        writer.add_custom_scalars(e2e_custom_scalars)

    if video:
        for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
            save_toss_video(writer, system, bp, data_manager.train_runs[i],
                            f'f_toss(red=data,blue=prediciton)/train{i}')
            save_toss_video(writer, system, bp, data_manager.valid_runs[i],
                            f'f_toss(red=data,blue=prediciton)/valid{i}')
