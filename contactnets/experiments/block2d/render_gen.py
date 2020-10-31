import sys
sys.path.append('..')

import pickle

import torch

from argparse import ArgumentParser

from contactnets.system import System
from contactnets.utils import utils, file_utils, dirs, system_io
from contactnets.vis import Visualizer2D

from contactnets.interaction import DirectResolver, DirectLearnable, PolyGeometry2D
from contactnets.experiments.block2d import sim, Block2DParams, StructuredLearnable, DeepLearnable 
from contactnets.experiments.block2d.train import Block2DTraining, Block2DTrainingE2E

import click

import pdb

@click.command()
@click.option('--num', default=0, help='Which run to visualize')
@click.option('--compare/--no-compare', default=False,
              help='Whether to render the compared model as well')
@click.option('--save/--no-save', default=False, help='Save into out/renders directory')
def main(num, compare, save):
    torch.set_default_tensor_type(torch.DoubleTensor)
    
    if compare:
        training = file_utils.load_params(None, 'training')
        structured = isinstance(training, Block2DTraining)
        bp: Block2DParams = file_utils.load_params(training.device, 'experiment')
    else:
        bp: Block2DParams = file_utils.load_params('cpu', 'experiment')

    x = torch.load(dirs.out_path('data', 'all', str(num) + '.pt'))

    system = sim.create_empty_system(bp=bp)
    system_io.load_system(x, system)
    sim_results = [system.get_sim_result()]

    if compare:
        if structured:
            interaction = StructuredLearnable(system.entities[0], system.entities[1],
                bp.vertices, bp.mu, training.H, training.learn_normal, training.learn_tangent)
            system.resolver.interactions = torch.nn.ModuleList([interaction])
        else:
            interaction = DeepLearnable(system.entities)
            system.resolver = DirectResolver(torch.nn.ModuleList([interaction]))

        system.load_state_dict(torch.load(dirs.out_path('trainer.pt')))
        system.eval()
        
        system.restart_sim()

        sim_results.append(system.get_sim_result())

    lcp = system.resolver
    
    vis = Visualizer2D(sim_results, [utils.create_geometry2d(bp.vertices)], system.params)
    vis.render(save_video=save)

if __name__ == "__main__": main()
