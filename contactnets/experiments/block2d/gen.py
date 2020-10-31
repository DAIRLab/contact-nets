import sys
sys.path.append('..')

import torch
from torch import Tensor

import math
import numpy as np
import numpy.random as rand

import pickle

from contactnets.system import System
from contactnets.utils import dirs, system_io, file_utils

from contactnets.experiments.block2d import sim, Block2DParams

from typing import *

import math

import click

import pdb

params = Block2DParams(dt=torch.tensor(0.05), mu=torch.tensor(0.8))

def create_random_system(step_n: int, random_configuration=True, random_velocity=True) -> System:
    configuration = Tensor([0, 3, 0])
    if random_configuration:
        rot = torch.empty(1).uniform_(0, 2 * math.pi)
        noise = torch.cat((torch.empty(2).uniform_(-0.3, 0.3), rot), dim=0)
        configuration = configuration + noise
    
    velocity = Tensor([0, -1, 0])
    if random_velocity:
        velocity = velocity + torch.empty(3).uniform_(-4, 4)
    
    system = sim.create_system(configuration, velocity, step_n=step_n, bp = params)
    return system

def create_controls(step_n: int, random=False) -> List[List[Tensor]]:
    control = Tensor([0, 0, 0])
    if random:
        control = control + torch.empty(3).uniform_(-4, 4)

    return sim.create_controls(control, step_n)

def random_toss_gen(run_n: int, step_n: int) -> None:
    params.run_n = run_n
    params.step_n = step_n
    file_utils.save_params(params, 'experiment')

    with click.progressbar(range(run_n)) as bar:
        for run in bar:
            system = create_random_system(step_n)
            controls = create_controls(step_n)

            results = system.sim(controls)
            
            x = system_io.serialize_system(system)

            torch.save(x, dirs.out_path('data', 'all', str(run) + '.pt'))

def do_gen(run_n: int, step_n: int) -> None:
    torch.set_default_tensor_type(torch.DoubleTensor)

    file_utils.create_empty_directory(dirs.out_path('data', 'all'))
    file_utils.clear_directory(dirs.out_path('data', 'train'))
    file_utils.clear_directory(dirs.out_path('data', 'valid'))
    file_utils.clear_directory(dirs.out_path('data', 'test'))

    random_toss_gen(run_n, step_n)

@click.command()
@click.option('--runs', default=60, help='Number of random runs to generate')
@click.option('--steps', default=50, help='Number of steps in each run')
def main(runs: int, steps: int):
    do_gen(runs, steps)

if __name__ == "__main__": main()
