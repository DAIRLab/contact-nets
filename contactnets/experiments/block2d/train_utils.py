import copy
import json
import math
from numbers import Number
import os
import os.path
import pdb  # noqa
from typing import TYPE_CHECKING, Dict, cast

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d  # noqa
import numpy as np
import skvideo.io
from tensorboardX import SummaryWriter
import torch
from torch import Tensor

from contactnets.entity import Ground2D
from contactnets.experiments.block2d import Block2DParams
from contactnets.experiments.block2d.structured_learnable import StructuredLearnable
from contactnets.interaction import PolyGeometry2D
from contactnets.system import System
from contactnets.train import DataManager
from contactnets.utils import dirs, file_utils, system_io
from contactnets.vis import Visualizer2D

if TYPE_CHECKING:
    from contactnets.experiments.block2d.train import Block2DTraining, Block2DTrainingE2E


def save_toss_video(writer: SummaryWriter, system: System,
                    bp: Block2DParams, run: Tensor, name: str) -> None:
    system_io.load_system(run, system)
    sim_results = [system.get_sim_result()]
    system.restart_sim()
    sim_results.append(system.get_sim_result())

    vis = Visualizer2D(sim_results, [PolyGeometry2D(bp.vertices)],
                       [cast(Ground2D, system.entities[1])], system.params)
    file_path = vis.render(save_video=True)

    if file_path is not None:
        video = skvideo.io.vread(file_path)
        # batch_n = 1 x time_steps x colors x H x W
        # Move color and add empty batch dim
        video = np.expand_dims(np.moveaxis(video, 3, 1), 0)
        # Downsample video
        video = video[:, :, :, 3::4, 3::4]

        writer.add_video(name, video, fps=15)

        os.remove(file_path)


def phi_render(interaction: StructuredLearnable, writer: SummaryWriter,
               normal=True, vrange=None) -> None:
    fig = plt.figure(figsize=(5, 5))
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
    xs_plt = xs_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n, points_n).T
    ths_plt = ths_orig.repeat((1, points_n)).reshape(-1, 1).reshape(points_n, points_n)

    if normal:
        coords = torch.cat((xs, ys, ths), dim=1)
        phi = interaction.phi_net(coords)
    else:
        coords = torch.cat((ys, xs, ths), dim=1)
        phi = interaction.tangent_net(coords)

    phi = torch.abs(phi.squeeze(1))
    n_cont = phi.shape[1]
    phi2 = phi.reshape((points_n ** 2, points_n, phi.shape[1]))
    best_ys = ys_orig[torch.min(phi2, dim=1).indices].squeeze(2)
    for plt_i in range(n_cont):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ys_plt_i = best_ys[:, plt_i].reshape(points_n, points_n).T

        def np_cast(t): return t.detach().cpu().numpy()
        surf = ax.plot_surface(np_cast(xs_plt), np_cast(ths_plt),
                               np_cast(ys_plt_i), cmap=cm.coolwarm, linewidth=0)
        ax.set_zlim(extent[2], extent[3])
        ax.set_xlabel('x' if normal else 'y')
        ax.set_ylabel('theta')
        ax.set_zlabel('y' if normal else 'x')
        ax.set_title(('phi(' if normal else 'phi_t') + str(plt_i) + ') = 0 surface')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        title_base = 'b_phi/ind_' if normal else 'd_phi_t/ind_'
        writer.add_figure(title_base + str(plt_i), fig)


def phi_render_slice(interaction: StructuredLearnable, writer: SummaryWriter,
                     vrange=None, normal=True) -> None:
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
        coords = torch.cat((torch.zeros(xsys.shape), xsys, thetas), dim=1)
    else:
        coords = torch.cat((xsys, torch.zeros(xsys.shape), thetas), dim=1)

    phi = interaction.phi_net(coords) if normal else interaction.tangent_net(coords)
    n_cont = phi.shape[1]

    for plt_i in range(n_cont):
        fig = plt.figure(figsize=(6, 5))

        phi_i = phi[:, plt_i].reshape((points_n, points_n)).t().flip(dims=[0])
        phi_i = phi_i.cpu().detach().numpy()

        phi_range = np.abs(phi_i).max()
        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])

        if vrange is None: vrange = [-phi_range, phi_range]

        plt.imshow(phi_i, vmin=vrange[0], vmax=vrange[1],
                   cmap=plt.cm.seismic, extent=extent, aspect=aspect)

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


ct = 'Multiline'
structured_custom_scalars = {
    'Prediction': {'Position':    [ct, ['train_pos_pred',         'valid_pos_pred']],
                   'Angle':       [ct, ['train_angle_pred',       'valid_angle_pred']],
                   'Velocity':    [ct, ['train_vel_pred',         'valid_vel_pred']],
                   'Angular Vel': [ct, ['train_angle_vel_pred',   'valid_angle_vel_pred']]},
    'Trajectory': {'Pos Final':   [ct, ['train_pos_final_traj',   'valid_pos_final_traj']],
                   'Angle Final': [ct, ['train_angle_final_traj', 'valid_angle_final_traj']],
                   'Pos Int':     [ct, ['train_pos_int_traj',     'valid_pos_int_traj']],
                   'Angle Int':   [ct, ['train_angle_int_traj',   'valid_angle_int_traj']],
                   'Pos Bound':   [ct, ['train_pos_bound_traj',   'valid_pos_bound_traj']],
                   'Angle Bound': [ct, ['train_angle_bound_traj', 'valid_angle_bound_traj']]},
    'Surrogate': {'Surrogate':    [ct, ['train_surrogate',        'valid_surrogate']]},
    'Time':      {'Time':         [ct, ['time_train',             'time_log']]}
}


def tensorboard_callback_structured(writer: SummaryWriter, epoch_vars: Dict[str, Number],
                                    training: 'Block2DTraining', bp: Block2DParams,
                                    system: System, data_manager: DataManager) -> None:

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

    for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
        save_toss_video(writer, system, bp, data_manager.train_runs[i], f'f_toss/train{i}')
        save_toss_video(writer, system, bp, data_manager.valid_runs[i], f'f_toss/valid{i}')


e2e_custom_scalars = {
    'Prediction': {'Position':    [ct, ['train_pos_pred',         'valid_pos_pred']],
                   'Angle':       [ct, ['train_angle_pred',       'valid_angle_pred']],
                   'Velocity':    [ct, ['train_vel_pred',         'valid_vel_pred']],
                   'Angular Vel': [ct, ['train_angle_vel_pred',   'valid_angle_vel_pred']]},
    'Trajectory': {'Pos Final':   [ct, ['train_pos_final_traj',   'valid_pos_final_traj']],
                   'Angle Final': [ct, ['train_angle_final_traj', 'valid_angle_final_traj']],
                   'Pos Int':     [ct, ['train_pos_int_traj',     'valid_pos_int_traj']],
                   'Angle Int':   [ct, ['train_angle_int_traj',   'valid_angle_int_traj']],
                   'Pos Bound':   [ct, ['train_pos_bound_traj',   'valid_pos_bound_traj']],
                   'Angle Bound': [ct, ['train_angle_bound_traj', 'valid_angle_bound_traj']]},
    'Velocity':  {'Vel Basic':    [ct, ['train_vel_basic',        'valid_vel_basic']]},
    'Time':      {'Time':         [ct, ['time_train',             'time_log']]}
}


def tensorboard_callback_e2e(writer: SummaryWriter, epoch_vars: Dict[str, Number],
                             training: 'Block2DTrainingE2E', bp: Block2DParams,
                             system: System, data_manager: DataManager) -> None:

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

    for i in range(min(4, len(data_manager.train_runs), len(data_manager.valid_runs))):
        save_toss_video(writer, system, bp, data_manager.train_runs[i], f'f_toss/train{i}')
        save_toss_video(writer, system, bp, data_manager.valid_runs[i], f'f_toss/valid{i}')
