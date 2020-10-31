import subprocess
import random

from contactnets.utils import dirs, file_utils
from contactnets.experiments import split
from contactnets.experiments.block3d import train
from contactnets.experiments.block3d.train import Block3DTraining

import time
import threading

import socket

import click

import pdb

@click.command()
@click.option('--method', type=click.Choice(['e2e', 'polytope', 'deep']),
        default='e2e', help='Which method to train with.')
@click.option('--tosses', default=100, help='Number of training tosses. Data split is 50% training, 30% validation, 20% test (so total number of tosses used will be 2x what is specified).')
def main(method: str, tosses: int):
    assert method is not None
    assert tosses <= 284, 'Number of training tosses must be less than half the dataset size'
    print(f'Executing method {method} with {tosses} training tosses')

    total_tosses = max(3, 2 * tosses)

    patience = int(500.0 / total_tosses) + 12

    split.do_split('50,30,20', num_tosses = total_tosses)

    if method == 'e2e':
        train.do_train_e2e(epochs=500, batch=1, patience=patience, resume=False)
    else:
        training = Block3DTraining()
        training.polytope = (method == 'polytope')
        train.do_train(epochs=500, batch=1, patience=patience, training=training, resume=False)

if __name__ == '__main__': main()
