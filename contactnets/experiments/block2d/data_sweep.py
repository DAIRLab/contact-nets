import os
import random
import subprocess
import threading
import time

import click

from contactnets.experiments import split
from contactnets.experiments.block2d import gen, train
from contactnets.utils import dirs, file_utils

# Make pygame renders headless
os.environ['SDL_VIDEODRIVER'] = 'dummy'


@click.command()
@click.option('--resume/--restart', default=False)
@click.option('--e2e/--structured', default=False)
def main(resume: bool, e2e: bool):
    command = 'tensorboard --samples_per_plugin images=0 --port 6007 --bind_all ' \
              '--logdir {} > /dev/null 2>&1'.format(dirs.results_path('sweep'))
    t = threading.Thread(target=os.system, args=(command,))
    t.start()

    run_nums = [5 * 2**j for j in range(1, 8)]
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

        gen_interrupt = os.path.exists(data_dir) and (file_utils.num_files(data_dir) != run_num)

        patience = int(200.0 / run_num) + 8
        scheduler_step_size = min(int(1800.0 / run_num) + 3, 50)

        def do_train(resume):
            args = {'epochs': 500, 'batch': 1, 'patience': patience,
                    'scheduler_step_size': scheduler_step_size, 'resume': resume}
            train.do_train_e2e(**args) if e2e else train.do_train_structured(**args)

        if i == 0 and resume and not gen_interrupt:
            print(f'Resuming run_num: {run_num}')
            do_train(True)
        else:
            if gen_interrupt and resume and i == 0:
                print('Got interrupted during generation, starting fresh')
            print(f'Starting run_num: {run_num} with patience {patience} '
                  f'and schedule step {scheduler_step_size}')
            gen.do_gen(run_num, 50)
            split.do_split('50,30,20')
            do_train(False)

        command_tokens = ['cp', '-R', dirs.out_path('.'), run_dir]
        print(subprocess.call(command_tokens))
        command_tokens = ['rsync', '-rv', '--exclude=tensorboard', dirs.out_path('.'),
                          dirs.results_path('sweep-no-tb', str(run_num))]
        print(subprocess.call(command_tokens, stdout=devnull, stderr=devnull))

        elapsed = time.time() - t0
        print(f'Took time: {elapsed}')


if __name__ == '__main__': main()
