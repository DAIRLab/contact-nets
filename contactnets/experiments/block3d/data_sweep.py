import os
# Make pygame renders headless
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import subprocess
import random

from contactnets.utils import dirs, file_utils
from contactnets.utils.processing import process_dynamics
from contactnets.experiments import split
from contactnets.experiments.block3d import gen, train

import time
import threading

import socket

import click

import pdb

@click.command()
@click.option('--resume/--restart', default=False)
@click.option('--generate/--data', default=True)
@click.option('--e2e/--surrogate', default=False)
@click.option('--deploy/--test', default=True)
def main(resume: bool, generate: bool, e2e: bool, deploy: bool):
    command = 'tensorboard --samples_per_plugin images=0 --port 6007 --bind_all --logdir {} --window_title {} > /dev/null 2>&1'.format(dirs.results_path('sweep'), socket.gethostname())
    t = threading.Thread(target=os.system, args=(command,))
    t.start()

    run_nums = [4 * 2**j for j in range(1, 8)]
    if deploy:
        random.shuffle(run_nums)

    if resume:
        for directory in os.listdir(dirs.results_path('sweep')):
            if file_utils.num_files(dirs.results_path('sweep', directory)) == 0:
                print(f'Detected partially completed: {directory}')
                run_nums.insert(0, run_nums.pop(run_nums.index(int(directory))))
            else:
                run_nums.remove(int(directory))
    else:
        file_utils.create_empty_directory(dirs.results_path('sweep'))
        file_utils.create_empty_directory(dirs.results_path('sweep-no-tb'))

    devnull = open(os.devnull, 'w')

    for i, run_num in enumerate(run_nums):
        t0 = time.time()

        run_dir = dirs.results_path('sweep', str(run_num))
        file_utils.create_empty_directory(run_dir)

        data_dir = dirs.out_path('data', 'all')

        gen_interrupt = os.path.exists(data_dir) and (file_utils.num_files(data_dir) < run_num)

        if deploy:
            patience = int(500.0 / run_num) + 12
        else:
            patience = 0
        scheduler_step_size = min(int(1800.0 / run_num) + 3, 50)

        def do_train(resume):
            train_file = dirs.root_path('contactnets', 'experiments', 'block3d', 'train.py')
            command = f'python3.7 {train_file} --patience {patience} --epochs 500'
            if resume: command += ' --resume' 
            if e2e: command += ' --e2e'
            os.system(command)
            # if e2e:
                # train.do_train_e2e(epochs=500, batch=1, patience=patience, resume=resume)
            # else:
                # train.do_train(epochs=500, batch=1, patience=patience, resume=resume)


        if i == 0 and resume and not gen_interrupt:
            print(f'Resuming run_num: {run_num}')
            do_train(True)
        else:
            if gen_interrupt and resume and i==0:
                print('Got interrupted during generation, starting fresh')
            print(f'Starting run_num: {run_num} with patience {patience} and schedule step {scheduler_step_size}')
            if generate:
                gen.do_gen(run_num, 50, gen.params, elastic=False, restitute_friction=False)
            else:
                process_dynamics.do_process_multi(run_num, downsample=1, center=True, perturb=True,
                                zrot=True, pullback=True, use_python3=False, plot=False, toss_comp=0,
                                zshift=0.0, length_scale=1 / process_dynamics.BLOCK_HALF_WIDTH)
            split.do_split('50,30,20')
            do_train(False)

        command = ['cp', '-R', dirs.out_path('.'), run_dir]
        print(subprocess.call(command))
        command = ['rsync', '-rv', '--exclude=tensorboard', dirs.out_path('.'), dirs.results_path('sweep-no-tb', str(run_num))]
        print(subprocess.call(command, stdout=devnull, stderr=devnull))

        elapsed = time.time() - t0
        print(f'Took time: {elapsed}')

if __name__ == '__main__': main()
