
import pdb  # noqa

import click
import torch

from contactnets.experiments.ballbin2d import BallBin2DParams, sim
from contactnets.experiments.ballbin2d.train import BallBin2DTraining
from contactnets.utils import dirs, file_utils, system_io
from contactnets.vis import Visualizer2D


@click.command()
@click.option('--num', default=0, help='Which run to visualize')
@click.option('--save/--no_save_traj', default=False, help='Save into out/renders directory')
def main(num: int, save: bool) -> None:
    torch.set_default_tensor_type(torch.DoubleTensor)

    training: BallBin2DTraining = BallBin2DTraining()
    bp: BallBin2DParams = file_utils.load_params(torch.device(training.device), 'experiment')

    x = torch.load(dirs.out_path('data', 'all', str(num) + '.pt'))

    system = sim.create_empty_system(bp=bp)
    system_io.load_system(x, system)
    sim_results = [system.get_sim_result()]

    lcp = system.resolver
    vis = Visualizer2D(sim_results, lcp.interactions[0].ball_geometries,
                       lcp.interactions[0].walls, system.params)
    vis.camera.y = 4
    vis.camera.zoom = 40.0
    vis.render(save_video=save)


if __name__ == "__main__": main()
