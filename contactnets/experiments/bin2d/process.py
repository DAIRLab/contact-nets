from argparse import ArgumentParser

import os

import torch
from torch import Tensor
import torch.distributions as tdist
from torch.utils.data import Dataset, TensorDataset, DataLoader

import pickle
import numpy as np
import random

from contactnets.experiments.block3d import Block3DParams
from contactnets.utils import dirs

from typing import *
import pdb

class DataProcessor:
    def __init__(self, device, length_scale=20.0, mass_scale=3.0, noise=0.0, subsample=False):
        self.device = device
        self.length_scale = length_scale
        self.mass_scale = mass_scale
        self.noise = noise
        self.subsample = subsample

    def load_data(self, run_aux=None):
        self.train_runs = self.load_runs('train', run_aux=run_aux)
        self.validation_runs = self.load_runs('validation', run_aux=run_aux) 
        self.train_data = torch.cat(self.process_runs(self.train_runs))
        self.validation_data = torch.cat(self.process_runs(self.validation_runs))

        self.train_dataset, self.validation_dataset = self.make_datasets(self.train_data, self.validation_data)

    def load_params(self) -> Block3DParams:
        with open(dirs.out_path('params'), 'rb') as file:
            params = pickle.load(file)

        # Send all tensors to appropriate device
        for field in params.__dict__:
            attr = getattr(params, field)
            if isinstance(attr, torch.Tensor):
                setattr(params, field, attr.to(self.device))
        
        params.mass = self.mass_scale * params.mass
        params.inertia = self.mass_scale * (self.length_scale**2) * params.inertia
        params.vertices = self.length_scale * params.vertices
        params.g = self.length_scale * params.g

        if self.subsample:
            params = params._replace(dt=params.dt * 2)
        
        return params 

    def make_datasets(self, train_data, validation_data) -> Tuple[Dataset, Dataset]:
        dist = tdist.Normal(torch.tensor([0.0]), torch.tensor([self.noise]))
        train_data = train_data + dist.sample(train_data.shape).squeeze(3)
        validation_data = validation_data + dist.sample(validation_data.shape).squeeze(3)
        
        train_dataset = TensorDataset(train_data, train_data)
        validation_dataset = TensorDataset(validation_data, validation_data)

        return train_dataset, validation_dataset

    def load_runs(self, subdir, run_aux=None) -> List[Tensor]:
        datas = []

        file_num = 0
        file = dirs.out_path('raw', subdir, str(file_num) + '.pt')
        while os.path.exists(file):
            data = torch.load(file).to(self.device)
            
            data = self.length_scale_data(data)
            
            if not (run_aux is None):
                data = torch.cat((data, run_aux(data)), dim=1)

            datas.append(data.unsqueeze(0))
            file_num = file_num + 1
            file = dirs.out_path('raw', subdir, str(file_num) + '.pt')
        
        # To see ground height data
        # pdb.set_trace()
        # zs = [data[:, 2] for data in datas]
        # zscat = torch.cat(zs)
        # zs_list = [z.item() for z in zscat]
        # zs_ground = list(filter(lambda z: z < 1.1, zs_list))

        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('tkagg')
        # plt.hist(zs_ground, bins=50)
        # plt.show()

        return datas

    def length_scale_data(self, data):
        data[:, 0:3] *= self.length_scale
        data[:, 7:10] *= self.length_scale
        data[:, 13:19] *= self.length_scale
        return data

    def process_runs(self, runs: List[Tensor]) -> List[Tensor]:
        return [self.process_run(run) for run in runs]

    def process_run(self, data: Tensor) -> Tensor:
        data = data.squeeze(0)

        # Turns time sequenced data into learning batch
        if self.subsample:
            indices = list(range(0, data.shape[1], 2))
            data = data[indices, :]

        
        n = data.shape[0]
        batch = torch.zeros((n-1, 2, data.shape[1]))

        for i in range(n-1):
            batch[i, :, :] = torch.cat((data[i, :].unsqueeze(0),
                                        data[i+1, :].unsqueeze(0)))
        return batch.to(self.device)
