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

from contactnets.experiments.block3d import sim, Block3DParams

from typing import *

import math

import click

import pdb

#run_n = 10
#step_n = 60

scale = 1.0

params = Block3DParams()
#params.g *= scale
params.inertia *= scale**2
params.vertices *= scale


def create_random_system(step_n: int, random_configuration=True, random_velocity=True, \
        bp: Block3DParams = Block3DParams(), elastic: bool = False, restitute_friction: bool = True) -> System:

    configuration = Tensor([0, 0, 2.1, 0, 0, 0, 0])
    if random_configuration:
        rot = torch.empty(1).uniform_(0, 2 * math.pi)

        vec = torch.empty(3).uniform_(-1, 1)
        vec = vec / torch.norm(vec, 2)
        quat = torch.cat((torch.cos(rot / 2), torch.sin(rot/2) * vec))

        noise = torch.cat((torch.empty(3).uniform_(-0.3, 0.3), quat), dim=0)
        noise[:2] *= 6
        noise[2] *= 0.0
        configuration = configuration + noise

    velocity = Tensor([0, 20, 0, 0, 0, 0])
    # velocity = Tensor([0, 8, 0, 0, 0, 0])
    #velocity = Tensor([0, 0, 0, 0, 0, 0])
    if random_velocity:
        VMAG = 2
        WMAG = 4
        #velocity = velocity + MAG * torch.empty(6).uniform_(-1, 1)
        velocity = velocity + torch.cat((VMAG * torch.empty(3).uniform_(-1, 1), WMAG * torch.empty(3).uniform_(-1, 1)), dim=0)
        #make it go downward, not getting enough data in contact
        velocity[2] = -0.5*torch.abs(velocity[2])
        #velocity[0] = 3.0*velocity[0]

    #pdb.set_trace()
    system = sim.create_system(configuration, velocity, step_n=step_n, bp=bp, elastic=elastic, restitute_friction=restitute_friction)
    return system

def create_controls(step_n: int, random=False) -> List[List[Tensor]]:
    control = Tensor([0, 0, 0, 0, 0, 0])
    if random:
        control = control + torch.empty(6).uniform_(-4, 4)

    return sim.create_controls(control, step_n)

def random_toss_gen(run_n: int, step_n: int, bp: Block3DParams, elastic: bool, restitute_friction: bool) -> None:
    bp.run_n = run_n
    bp.step_n = step_n
    file_utils.save_params(bp, 'experiment')
    #system = create_random_system(step_n, bp=bp, elastic=elastic)

    #controls = create_controls(step_n)
    #results = system.sim(controls)
    #pdb.set_trace()
    with click.progressbar(range(run_n)) as bar:
        for run in bar:
            #pdb.set_trace()
            system = create_random_system(step_n, bp=bp, elastic=elastic, restitute_friction=restitute_friction)

            controls = create_controls(step_n)
            N = 10
            while True:
                try:
                    results = system.sim(controls)
                    break
                except:

                    N = N - 1
                    if N < 0:
                        raise
                    system = create_random_system(step_n, bp=bp, elastic=elastic, restitute_friction=restitute_friction)
                    controls = create_controls(step_n)

            x = system_io.serialize_system(system)

            torch.save(x, dirs.out_path('data', 'all', str(run) + '.pt'))


def do_gen(run_n: int, step_n: int, bp: Block3DParams, elastic: bool, restitute_friction: bool) -> None:
    torch.set_default_tensor_type(torch.DoubleTensor)

    file_utils.create_empty_directory(dirs.out_path('data', 'all'))
    file_utils.clear_directory(dirs.out_path('data', 'train'))
    file_utils.clear_directory(dirs.out_path('data', 'valid'))
    file_utils.clear_directory(dirs.out_path('data', 'test'))

    random_toss_gen(run_n, step_n, bp=bp, elastic=elastic, restitute_friction=restitute_friction)

@click.command()
@click.option('--runs', default=60, help='Number of random runs to generate')
@click.option('--steps', default=80, help='Number of steps in each run')
@click.option('--elastic/--inelastic', default=False)
@click.option('--restitute_friction/--restitute_normal', default=True)
def main(runs: int, steps: int, elastic: bool, restitute_friction: bool):
    do_gen(runs, steps, params, elastic, restitute_friction)

if __name__ == "__main__": main()
