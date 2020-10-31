from tensorboardX import SummaryWriter

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

import os, subprocess
import numpy as np
from numbers import Number

from typing import *

from contactnets.utils import utils, dirs, file_utils
from contactnets.jacnet import Series

import pdb

class DataManager:
    train_runs: List[Tensor]
    valid_runs: List[Tensor]
    test_runs: List[Tensor]

    train_dataset: Dataset
    valid_dataset: Dataset
    test_dataset: Dataset
    device: torch.device

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def load(self, noise=0.0) -> None:
        def process_run(data: Tensor) -> Tensor:
            """
            Turns time sequenced data into a batch of paired time steps.
            data has shape      1 x n_steps x [state control]s
            gets transformed to n_steps-1 x 2 x [state control]s
            """
            assert(data.shape[0] == 1)

            n = data.shape[1]
            batch = torch.zeros((n-1, 2, data.shape[2]))

            for i in range(n-1):
                batch[i, :, :] = torch.cat((data[0, i, :].unsqueeze(0),
                                            data[0, i+1, :].unsqueeze(0)))
            return batch.to(self.device)
        
        def load_runs(subdir: str) -> List[Tensor]:
            datas = []
            
            try:
                for file_name in os.listdir(dirs.out_path('data', subdir)):
                    file_path = dirs.out_path('data', subdir, file_name)
                    data = torch.load(file_path).to(self.device)
                    data = data + torch.randn(data.shape) * noise
                    datas.append(data)
            except:
                print(f'Could not load data for {subdir}, did you run experiment/split.py?')

            return datas
        
        self.train_runs = load_runs('train')
        self.valid_runs = load_runs('valid')
        self.test_runs  = load_runs('test')
        
        if len(self.train_runs) == len(self.valid_runs) == len(self.test_runs) == 0:
            raise RuntimeError('No data found; did you run split.py after gen.py?')

        train_processed = list(map(process_run, self.train_runs))
        valid_processed = list(map(process_run, self.valid_runs))
        test_processed  = list(map(process_run, self.test_runs))
        
        if self.train_runs: self.train_dataset = TensorDataset(torch.cat(train_processed))
        else: self.train_dataset = Dataset()

        if self.valid_runs: self.valid_dataset = TensorDataset(torch.cat(valid_processed))
        else: self.valid_dataset = Dataset()

        if self.test_runs:  self.test_dataset = TensorDataset(torch.cat(test_processed))
        else: self.test_dataset = Dataset()
