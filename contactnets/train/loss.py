import torch
from torch import Tensor
from torch.nn import Module

from typing import *
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import random

from contactnets.system import System
from contactnets.utils import utils, system_io, file_utils, dirs
from contactnets.train.data_manager import DataManager

import pdb

class LossType(Enum):
    STEPWISE = 1
    TRAJECTORY = 2

class Loss(ABC):
    names: List[str] # One for each entry returned by compute_loss
    system: System
    loss_type: LossType
    mutates: bool

    def __init__(self, names: List[str], system: System,
                       loss_type: LossType, mutates: bool) -> None:
        self.names = names
        self.system = system
        self.loss_type = loss_type
        self.mutates = mutates

    @abstractmethod
    def compute_loss(self, meta_save_file: str = None) -> List[Tensor]:
        """
        Computes a loss over the states in the underlying System.
        System may be modified if the loss mutates.
        Can return multiple loss metrics for computation efficiency.
        """
        pass

    def preprocess_data(self, data) -> Tensor:
        return data

@dataclass
class ReportingLoss:
    loss: Loss
    sample_n: int

class LossManager:
    system: System
    training_losses: List[Loss]
    reporting_losses: List[ReportingLoss]

    def __init__(self, system: System,
                       training_losses: List[Loss],
                       reporting_losses: List[ReportingLoss]) -> None:
        self.system = system
        self.training_losses = training_losses
        self.reporting_losses = reporting_losses

    def compute_training_loss(self, stepwise_data: Tensor) -> Tensor:
        # data is of dimension batch_n x 2 x [state control]s
        system_io.load_system(stepwise_data, self.system)

        for i, loss in enumerate(self.training_losses):
            if len(self.training_losses) > 1: assert not loss.mutates
            assert loss.loss_type == LossType.STEPWISE

            new_loss = loss.compute_loss()[0]

            cum_loss = new_loss if i == 0 else cum_loss + new_loss

        return cum_loss

    def compute_reporting_losses(self, data: DataManager, types=['train','valid'],
            eval_all=False) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        loss_stats = {}
        losses = {}

        if eval_all:
            file_utils.create_empty_directory(dirs.out_path('best', 'meta'))

        def compute(reporting, datasets):
            for data_type, dataset in zip(['train', 'valid', 'test'], datasets):
                if data_type in types:
                    reporting_losses = self._reporting_loss(reporting, dataset, eval_all=eval_all, data_type=data_type)
                    names = reporting.loss.names
                    assert len(reporting_losses) == len(names)
                    for reporting_loss, name in zip(reporting_losses, names):
                        loss_stats[f'{data_type}_{name}'], losses[f'{data_type}_{name}'] = \
                           reporting_loss[0], reporting_loss[1]

        with torch.no_grad():
            for reporting in self.reporting_losses:
                if reporting.loss.loss_type == LossType.STEPWISE:
                    compute(reporting, [data.train_dataset, data.valid_dataset, data.test_dataset])
                elif reporting.loss.loss_type == LossType.TRAJECTORY:
                    compute(reporting, [data.train_runs, data.valid_runs, data.test_runs])

        return loss_stats, losses

    def _reporting_loss(self, loss: ReportingLoss, dataset, eval_all=False, data_type=None) \
            -> List[Tuple[float, List[float]]]:
        """
        dataset just needs to implement len() and __getitem__().
        The type of dataset should match the type of the loss (i.e. STEPWISE or TRAJECTORY)
        """

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        if not eval_all:
            indices = indices[:min(loss.sample_n, len(indices))]

        losses = None

        if eval_all:
            save_dir = dirs.out_path('best', 'meta', loss.loss.names[0], data_type)
            file_utils.create_empty_directory(save_dir)

        for i in indices:
            data = dataset[i]
            if type(data) is tuple:
                assert len(data) == 1
                data = data[0].unsqueeze(0)
            data = loss.loss.preprocess_data(data)
            system_io.load_system(data, self.system)
            
            if eval_all:
                meta_save_file = dirs.out_path('best', 'meta', loss.loss.names[0], data_type, f'{i}.pt')
                new_losses = loss.loss.compute_loss(meta_save_file=meta_save_file)
            else:
                new_losses = loss.loss.compute_loss()

            if losses is None:
                losses = [[] for i in range(len(new_losses))]

            for i, new_loss in enumerate(new_losses):
                losses[i].append(new_loss.item())

        reporting_losses = [(sum(single_losses) / len(indices), single_losses)
                            for single_losses in losses]

        return reporting_losses
