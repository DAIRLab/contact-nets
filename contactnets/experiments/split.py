import random
import shutil
from typing import List

import click
import numpy as np

from contactnets.utils import dirs, file_utils


def move_runs(to_dir: str, runs: List[int]) -> None:
    for run in runs:
        shutil.copyfile(dirs.out_path('data', 'all', f'{run}.pt'),
                        dirs.out_path('data', to_dir, f'{run}.pt'))


def do_split(split: str, num_tosses: int = None) -> None:
    splits = list([int(split) for split in split.split(',')])

    assert sum(splits) == 100, 'Percentages must add to 100'

    file_utils.create_empty_directory(dirs.out_path('data', 'train'))
    file_utils.create_empty_directory(dirs.out_path('data', 'valid'))
    file_utils.create_empty_directory(dirs.out_path('data', 'test'))

    run_n = file_utils.num_files(dirs.out_path('data', 'all'))

    runs = list(range(run_n))
    random.shuffle(runs)

    if num_tosses is not None:
        runs = runs[:num_tosses]

    split_points = np.cumsum(splits)
    split_points = [int(np.floor(split * 0.01 * len(runs))) for split in split_points]

    train_data = runs[0:split_points[0]]
    valid_data = runs[split_points[0]:split_points[1]]
    test_data = runs[split_points[1]:]

    move_runs('train', train_data)
    move_runs('valid', valid_data)
    move_runs('test', test_data)

    print(f'Split data into train, valid, and test with split {split}')


@click.command(help='Takes data in out/data/raw and splits it randomly into'
                    'out/data/train and out/data/validation and out/data/test.')
@click.option('--split', default='50,30,20', show_default=True,
              help='Train, validation, test data split. Percentages must add up to 100.')
@click.option('--num_tosses', type=int, default=None, show_default=True,
              help='Specify to only use a random subset of tosses in out/data/all.')
def main(split: str, num_tosses: int) -> None:
    do_split(split, num_tosses)


if __name__ == "__main__": main()
