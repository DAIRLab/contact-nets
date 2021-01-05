import os
import pdb  # noqa
import unittest
import json

import torchtestcase as ttc

from contactnets.experiments.block3d.gen import do_gen as block3d_gen
from contactnets.experiments.block3d.train import do_train_structured as block3d_train_struct
from contactnets.experiments.block3d.train import do_train_e2e as block3d_train_e2e
from contactnets.experiments.block3d.train import Block3DTraining


from contactnets.experiments.block2d.gen import do_gen as block2d_gen
from contactnets.experiments.block2d.train import do_train_structured as block2d_train_struct
from contactnets.experiments.block2d.train import do_train_e2e as block2d_train_e2e
from contactnets.experiments.block2d.train import Block2DTraining

from contactnets.experiments.ballbin2d.gen import do_gen as ballbin_gen
from contactnets.experiments.ballbin2d.train import do_train_structured as ballbin_train_struct
from contactnets.experiments.ballbin2d.train import BallBin2DTraining


from contactnets.experiments.split import do_split as split
from contactnets.utils import dirs, file_utils


# Stop pygame from rendering
os.environ['SDL_VIDEODRIVER'] = 'dummy'


class TestIntegration(ttc.TorchTestCase):
    def setUp(self):
        file_utils.clear_directory(dirs.out_path())


    def test_ballbin2d(self):
        print('*********** TESTING BALLBIN2D **********')
        run_n, step_n, epochs = 3, 5, 2

        print('*********** GENERATING **********')
        ballbin_gen(run_n, step_n)
        self.split_and_assert(run_n)

        args = {'epochs': epochs, 'batch': 1, 'patience': 5, 'resume': False}

        print('*********** TRAINING STRUCTURED POLY **********')
        ballbin_train_struct(**args, training=BallBin2DTraining(polytope=True))
        self.assert_epochs_trained(epochs)

        print('*********** TRAINING STRUCTURED DEEP **********')
        ballbin_train_struct(**args, training=BallBin2DTraining(polytope=False))
        self.assert_epochs_trained(epochs)


    def test_block2d(self):
        print('*********** TESTING BLOCK2D **********')
        run_n, step_n, epochs = 3, 5, 2

        print('*********** GENERATING **********')
        block2d_gen(run_n, step_n)
        self.split_and_assert(run_n)

        args = {'epochs': epochs, 'batch': 1, 'patience': 5, 'scheduler_step_size': 5,
                'resume': False}

        print('*********** TRAINING STRUCTURED POLY **********')
        block2d_train_struct(**args, training=Block2DTraining(net_type='poly'))
        self.assert_epochs_trained(epochs)

        print('*********** TRAINING STRUCTURED DEEP **********')
        block2d_train_struct(**args, training=Block2DTraining(net_type='deep'))
        self.assert_epochs_trained(epochs)

        print('*********** TRAINING END TO END **********')
        block2d_train_e2e(**args)
        self.assert_epochs_trained(epochs)


    def test_block3d(self):
        print('*********** TESTING BLOCK3D **********')
        run_n, step_n, epochs = 3, 5, 2

        print('*********** GENERATING **********')
        block3d_gen(run_n, step_n)
        self.split_and_assert(run_n)

        args = {'epochs': epochs, 'batch': 1, 'patience': 5, 'resume': False}

        print('*********** TRAINING STRUCTURED POLY **********')
        block3d_train_struct(**args, training=Block3DTraining(net_type='poly'))
        self.assert_epochs_trained(epochs)

        print('*********** TRAINING STRUCTURED DEEP **********')
        block3d_train_struct(**args, training=Block3DTraining(net_type='deep'))
        self.assert_epochs_trained(epochs)

        print('*********** TRAINING STRUCTURED DEEPVERTEX **********')
        block3d_train_struct(**args, training=Block3DTraining(net_type='deepvertex'))
        self.assert_epochs_trained(epochs)

        print('*********** TRAINING END TO END **********')
        block3d_train_e2e(**args)
        self.assert_epochs_trained(epochs)

    def split_and_assert(self, run_n: int):
        self.assertEqual(file_utils.num_files(dirs.out_path('data', 'all')), run_n)

        split('34,33,33')
        self.assertEqual(file_utils.num_files(dirs.out_path('data', 'train')), run_n // 3)
        self.assertEqual(file_utils.num_files(dirs.out_path('data', 'valid')), run_n // 3)
        self.assertEqual(file_utils.num_files(dirs.out_path('data', 'test')), run_n // 3)

    def assert_epochs_trained(self, epochs: int):
        with open(dirs.out_path('variables.json')) as variables_file:
            variables = json.load(variables_file)
        self.assertEqual(len(variables), epochs + 1)


if __name__ == '__main__': unittest.main(buffer=True)
