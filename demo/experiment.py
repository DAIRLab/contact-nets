"""Provide a simple way to run different Block3D methods."""

import distutils.dir_util
import pdb  # noqa

import click

from contactnets.experiments import split
from contactnets.experiments.block3d import train
from contactnets.experiments.block3d.train import Block3DTraining
from contactnets.utils import dirs


@click.command()
@click.option('--method', type=click.Choice(['e2e', 'poly', 'deep', 'deepvertex']),
              default='e2e', help='Which method to train with.')
@click.option('--tosses', default=100,
              help='Number of training tosses. Data split is 50% training, 30% validation,'
                   '20% test (so total number of tosses used will be 2x what is specified).')
def main(method: str, tosses: int):
    assert method is not None
    assert tosses <= 284, 'Number of training tosses must be less than half the dataset size'

    # Copy the tosses data and processing parameters into the working directory
    distutils.dir_util.copy_tree(dirs.data_path('tosses_processed'),
                                 dirs.out_path('data', 'all'))
    distutils.dir_util.copy_tree(dirs.data_path('params_processed'),
                                 dirs.out_path('params'))

    epochs = 500
    total_tosses = max(3, 2 * tosses)
    patience = int(500.0 / total_tosses) + 12

    print(f'Executing method {method} with training tosses={tosses}, '
          f'patience={patience}, and epochs={epochs}')

    split.do_split('50,30,20', num_tosses = total_tosses)

    args = {'epochs': epochs, 'batch': 1, 'patience': patience, 'resume': False}

    if method == 'e2e':
        train.do_train_e2e(**args)  # type: ignore
    else:
        training = Block3DTraining(net_type=method)
        train.do_train_structured(**args, training=training)  # type: ignore


if __name__ == '__main__': main()
