import pdb  # noqa
from typing import Union, cast

import click
import torch

from contactnets.entity import Ground2D
from contactnets.experiments.block2d import (Block2DParams, DeepLearnable, StructuredLearnable,
                                             sim)
from contactnets.experiments.block2d.train import Block2DTraining, Block2DTrainingE2E
from contactnets.interaction import DirectResolver, PolyGeometry2D
from contactnets.utils import dirs, file_utils, system_io
from contactnets.vis import Visualizer2D


@click.command()
@click.option('--num', default=0, help='Which run to visualize')
@click.option('--compare/--no-compare', default=False,
              help='Whether to render the compared model as well')
@click.option('--save/--no-save', default=False, help='Save into out/renders directory')
def main(num: int, compare: bool, save: bool) -> None:
    torch.set_default_tensor_type(torch.DoubleTensor)

    if compare:
        training = file_utils.load_params(torch.device('cpu'), 'training')
        structured = isinstance(training, Block2DTraining)
        e2e        = isinstance(training, Block2DTrainingE2E)
        bp: Block2DParams = file_utils.load_params(torch.device(training.device), 'experiment')
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
            interaction = DeepLearnable(list(system.entities))
            system.resolver = DirectResolver([interaction])
        else:
            raise Exception("Don't recognize training file type")

        system.load_state_dict(torch.load(dirs.out_path('trainer.pt')))
        system.eval()

        system.restart_sim()

        sim_results.append(system.get_sim_result())

    vis = Visualizer2D(sim_results, [PolyGeometry2D(bp.vertices)],
                       [cast(Ground2D, system.entities[1])], system.params)
    vis.render(save_video=save)


if __name__ == "__main__": main()
