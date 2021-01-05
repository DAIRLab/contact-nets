import math
import pdb  # noqa
from typing import List

import click
import torch
from torch import Tensor

from contactnets.experiments.ballbin2d import BallBin2DParams, sim
from contactnets.system import System
from contactnets.utils import dirs, file_utils, system_io

torch.set_default_tensor_type(torch.DoubleTensor)


def create_random_system(step_n: int, random_configuration=True, random_velocity=True,
                         bp: BallBin2DParams = BallBin2DParams()) -> System:
    configurations = []
    velocities = []
    heights = bp.radii.cumsum(dim=0) * 2
    for (radius, height) in zip(bp.radii, heights):
        configuration = Tensor([0, 1 + height, 0])
        if random_configuration:
            rot = torch.empty(1).uniform_(0, 2 * math.pi)

            noise = torch.cat((torch.empty(1).uniform_(-radius, radius), torch.zeros(1), rot),
                              dim=0)
            configuration = configuration + noise

        velocity = Tensor([0, 0, 0])
        if random_velocity:
            VMAG = 2
            # velocity = velocity + MAG * torch.empty(6).uniform_(-1, 1)
            velocity = velocity + VMAG * torch.empty(3).uniform_(-1, 1)
            # make it go downward, not getting enough data in contact
            velocity[2] = -0.5 * torch.abs(velocity[2])
            # velocity[0] = 3.0*velocity[0]
        configurations += [configuration]
        velocities += [velocity]

    system = sim.create_system(configurations, velocities, step_n=step_n, bp=bp)
    return system


def create_controls(step_n: int, ball_n: int, wall_n: int, random=False) -> List[List[Tensor]]:
    control = Tensor([0, 0, 0])
    if random:
        control = control + torch.empty(3).uniform_(-4, 4)

    return sim.create_controls(control, step_n, ball_n, wall_n)


def random_toss_gen(run_n: int, step_n: int, bp: BallBin2DParams) -> None:
    bp.run_n = run_n
    bp.step_n = step_n
    file_utils.save_params(bp, 'experiment')

    with click.progressbar(range(run_n)) as bar:
        for run in bar:
            for _ in range(10):
                system = create_random_system(step_n, bp=bp)
                controls = create_controls(step_n, len(bp.radii), len(bp.angles))
                try:
                    system.sim(controls)
                    break
                except Exception:
                    pdb.set_trace()
                    continue
            else:
                raise Exception("Can't simulate system")

            x = system_io.serialize_system(system)
            torch.save(x, dirs.out_path('data', 'all', str(run) + '.pt'))


def do_gen(run_n: int, step_n: int, bp: BallBin2DParams = BallBin2DParams()) -> None:
    file_utils.create_empty_directory(dirs.out_path('data', 'all'))
    file_utils.clear_directory(dirs.out_path('data', 'train'))
    file_utils.clear_directory(dirs.out_path('data', 'valid'))
    file_utils.clear_directory(dirs.out_path('data', 'test'))

    random_toss_gen(run_n, step_n, bp=bp)


@click.command()
@click.option('--runs', default=60, help='Number of random runs to generate')
@click.option('--steps', default=200, help='Number of steps in each run')
def main(runs: int, steps: int):
    scale = 1.0

    params = BallBin2DParams()
    params.g *= scale
    params.inertias *= scale**2
    params.radii *= scale
    do_gen(runs, steps, params)


if __name__ == "__main__": main()
