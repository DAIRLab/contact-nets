import sys
sys.path.append('..')

import pickle

import torch
import numpy as np

from argparse import ArgumentParser

from contactnets.system import System
from contactnets.utils import file_utils, dirs, system_io
from contactnets.utils.processing import process_dynamics
from contactnets.vis import Visualizer3D

from contactnets.experiments.block3d import sim, Block3DParams, StructuredLearnable
from contactnets.experiments.block3d.train import Block3DTraining

import click

import pdb

@click.command()
@click.option('--num', default=0, help='Which run to visualize')
@click.option('--compare/--no_compare', default=False,
              help='Whether to render the compared model as well')
@click.option('--save/--no_save_traj', default=False, help='Save into out/renders directory')
@click.option('--save_traj/--no_save_traj', default=False, help='Save trajectory as csv')
def main(num, compare, save, save_traj):
    torch.set_default_tensor_type(torch.DoubleTensor)

    training: Block3DTraining = Block3DTraining()#file_utils.load_params(None, 'training')
    bp: Block3DParams = file_utils.load_params(training.device, 'experiment')

    x = torch.load(dirs.out_path('data', 'all', str(num) + '.pt'))

    system = sim.create_empty_system(bp=bp)
    system_io.load_system(x, system)
    sim_results = [system.get_sim_result()]

    if compare:
        dynamic = system.entities[0]
        system = sim.create_system(dynamic.configuration_history[0],
                                   dynamic.velocity_history[0],
                                   step_n=system.step_n(), bp=bp, elastic=training.elastic)

        interaction = StructuredLearnable(system.entities[0], system.entities[1],
                bp.vertices, bp.mu, training.H, training.learn_normal, training.learn_tangent)
        system.resolver.interactions = torch.nn.ModuleList([interaction])

        system.load_state_dict(torch.load(dirs.out_path('trainer.pt')))
        system.eval()

        system.sim(system_io.get_controls(x, system))

        sim_results.append(system.get_sim_result())

        if save_traj:
            xs_model_scaled = system_io.serialize_system(system).squeeze(0)
            
            xs_model_scaled[:, 0:3] *= process_dynamics.BLOCK_HALF_WIDTH

            t_start = torch.load(dirs.out_path('data', 'all', str(num) + '.pt.time')).item()
            times = t_start + np.arange(0, xs_model_scaled.shape[0], 1) * system.params.dt.item()
            times = np.expand_dims(times, 1)
            
            time_xs_model_scaled = np.concatenate((times, xs_model_scaled.detach()), axis=1)
            np.savetxt('/home/samuel/Repositories/tagslam_root/src/tagslam_video/sequences/foo/traj_1.csv', time_xs_model_scaled[:, 0:8], fmt='%1.8f') 

    lcp = system.resolver

    vis = Visualizer3D(sim_results, [lcp.interactions[0].geometry], system.params)
    vis.render(box_width=2.0, save_video=save)
    # vis.render(box_width=2.0, save_video=save, headless=save)


if __name__ == "__main__": main()
