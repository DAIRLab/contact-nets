import sys
sys.path.append('..')

import os
import os.path

import json

import json

import math
import numpy as np

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import skvideo.io

import time

import subprocess

from argparse import ArgumentParser

from tensorboardX import SummaryWriter
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from contactnets.system import System
from contactnets.train import Trainer, DataManager, TensorboardManager, TensorboardPlot, TensorboardSeriesHistogram, PolyGround3DSurrogate, PolyGround3DConeSurrogate, SurrogateConfig3D, ConeSurrogateConfig3D, Loss, ReportingLoss, LossManager, PredictionLoss, PredictionMode, TrajectoryLoss, BasicVelocityLoss
from contactnets.interaction import PolyGeometry3D, PolyGround3D, DirectResolver, DirectLearnable
from contactnets.vis import Visualizer3D

from contactnets.experiments.block3d import sim, Block3DParams, DeepLearnable, train

from contactnets.utils import file_utils, utils, dirs, system_io
from contactnets.utils import quaternion as quat

from typing import *
from numbers import Number
from dataclasses import dataclass

import click

import pdb

def do_sysid():
    training = train.Block3DTraining()

    device = torch.device(training.device if torch.cuda.is_available() else 'cpu')
    if training.device == 'cpu':
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    data_manager = DataManager(device)
    data_manager.load(noise=training.noise)
    hist_data = torch.cat(data_manager.train_runs, dim=1)
    hist_data = hist_data[:, :, 0:7]

    bp: Block3DParams = file_utils.load_params(device, 'experiment')

    MU_MIN, MU_N, MU_MAX = 0.1, 5, 0.2
    R_MIN, R_N, R_MAX = 0.0, 5, 0.25
    MUS = torch.linspace(MU_MIN, MU_MAX, MU_N)
    RS = torch.linspace(R_MIN, R_MAX, R_N)
    POS_PERF = torch.zeros(MU_N,R_N)
    ROT_PERF = torch.zeros(MU_N,R_N)

    for mu in range(MU_N):
        for r in range(R_N):
            #print(MUS)
            print('Running Settings: ', MUS[mu], RS[r])
            # Modify default system with a learnable polygon-ground interaction
            bp.restitution = RS[r]
            bp.mu = MUS[mu]

            system = sim.create_empty_system(bp, elastic=training.elastic)
            interaction = PolyGround3D(system.entities[0], system.entities[1],
                    bp.vertices, bp.mu)
            system.resolver.interactions = torch.nn.ModuleList([interaction])

            prediction_loss = PredictionLoss(system, mode=PredictionMode.XYZ_QUAT)
            trajectory_loss = TrajectoryLoss(system, prediction_loss)

            #if socp:
            reporting_losses = [ReportingLoss(trajectory_loss, 1000)]
            loss_manager = LossManager(system, [prediction_loss], reporting_losses)

            (a, b) = loss_manager.compute_reporting_losses(data_manager, types=['train'], eval_all=True)
            ROT_PERF[mu,r] = a['train_angle_int_traj']
            POS_PERF[mu,r] = a['train_pos_int_traj']
            print(ROT_PERF)
            print(POS_PERF)
    pdb.set_trace()



@click.command()

def main():
    do_sysid()

if __name__ == "__main__": main()
