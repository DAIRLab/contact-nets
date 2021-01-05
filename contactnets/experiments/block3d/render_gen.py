import pdb  # noqa
from typing import Union

import click
import numpy as np
import torch

from contactnets.experiments.block3d import (Block3DParams, DeepLearnable, StructuredLearnable,
                                             sim)
from contactnets.experiments.block3d.train import Block3DTraining, Block3DTrainingE2E
from contactnets.interaction import DirectResolver, PolyGeometry3D
from contactnets.system import System
from contactnets.utils import dirs, file_utils, system_io
from contactnets.utils.processing import process_dynamics
from contactnets.vis import Visualizer3D


def save_trajectory(system: System, num: int) -> None:
    xs_model_scaled = system_io.serialize_system(system).squeeze(0)

    xs_model_scaled[:, 0:3] *= process_dynamics.BLOCK_HALF_WIDTH

    t_start = torch.load(dirs.out_path('data', 'all', str(num) + '.pt.time')).item()
    dt = system.params.dt.item()
    times = t_start + np.arange(0, xs_model_scaled.shape[0], 1) * dt
    times = np.expand_dims(times, 1)

    time_xs_model_scaled = np.concatenate((times, xs_model_scaled.detach()), axis=1)
    np.savetxt('/home/samuel/Repositories/tagslam_root/src/tagslam_video/'
               'sequences/foo/traj_1.csv', time_xs_model_scaled[:, 0:8], fmt='%1.8f')


@click.command()
@click.option('--num', default=0, help='Which run to visualize')
@click.option('--compare/--no_compare', default=False,
              help='Whether to render the compared model as well')
@click.option('--save/--no_save_traj', default=False, help='Save into out/renders directory')
@click.option('--save_traj/--no_save_traj', default=False,
              help='Save trajectory as csv. The processing script needs'
                   'to have saved time stamps.')
def main(num: int, compare: bool, save: bool, save_traj: bool) -> None:
    torch.set_default_tensor_type(torch.DoubleTensor)

    if compare:
        training = file_utils.load_params(torch.device('cpu'), 'training')
        structured = isinstance(training, Block3DTraining)
        e2e        = isinstance(training, Block3DTrainingE2E)
        bp: Block3DParams = file_utils.load_params(torch.device(training.device), 'experiment')
    else:
        bp = file_utils.load_params(torch.device('cpu'), 'experiment')

    x = torch.load(dirs.out_path('data', 'all', str(num) + '.pt'))

    system = sim.create_empty_system(bp=bp)
    system_io.load_system(x, system)
    sim_results = [system.get_sim_result()]

    if compare:
        if structured:
            interaction: Union[StructuredLearnable, DeepLearnable] = \
                StructuredLearnable(system.entities[0], system.entities[1], bp.vertices,
                                    bp.mu, training.net_type, training.H,
                                    training.learn_normal, training.learn_tangent)
            system.resolver.interactions = torch.nn.ModuleList([interaction])
        elif e2e:
            interaction = DeepLearnable(list(system.entities),
                                        H=training.H, depth=training.depth)
            system.resolver = DirectResolver([interaction])
        else:
            raise Exception("Don't recognize training file type")

        system.load_state_dict(torch.load(dirs.out_path('trainer.pt')))
        system.eval()

        system.restart_sim()

        sim_results.append(system.get_sim_result())

        if save_traj: save_trajectory(system, num)

    vis = Visualizer3D(sim_results, [PolyGeometry3D(bp.vertices)], system.params)
    vis.render(box_width=2.0, save_video=save)


if __name__ == "__main__": main()
