import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, Dataset, DataLoader

import os
import sys
import subprocess
import logging
import time
import pickle, json
import traceback

from dataclasses import dataclass
from typing import *
from numbers import Number
import math

from contactnets.system import System
from contactnets.utils import utils, system_io, file_utils, dirs
from contactnets.train import Loss, LossManager, DataManager, TensorboardManager

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Trainer:
    system: System
    loss_manager: LossManager
    tensorboard_manager: TensorboardManager

    # Pairs of the full epoch variable with the shortened print tag
    # i.e. ('train_prediction', 'trainl2')
    epoch_prints: List[Tuple[str,str]]

    # Dictionary of important variables for every time step (l2 loss, etc)
    variables: List[Dict[str, Number]]

    optimizer_factor: Callable[[Iterator[Parameter]], Optimizer]
    scheduler_factory: Callable[[Optimizer], _LRScheduler]
    variables_adder: Callable[[], Dict[str, Number]]
    pre_step: Callable[[], None]
    post_step: Callable[[], None]

    def __init__(self, system: System,
                       data_manager: DataManager,
                       loss_manager: LossManager,
                       tensorboard_manager: TensorboardManager,
                       epoch_prints: List[Tuple[str,str]] = [],
                       optimizer_factory: Callable[[Iterator[Parameter]], Optimizer] = None,
                       scheduler_factory: Callable[[Optimizer], _LRScheduler] = None,
                       variables_adder: Callable[[], Dict[str, Number]] = None,
                       pre_step: Callable[[], None] = None,
                       post_step: Callable[[], None] = None) -> None:
        """System should contain things which can be learned; e.g., a inertia parameter
        in an Entity or a phi network in an Interaction."""
        super(Trainer, self).__init__()

        self.system = system
        self.data_manager = data_manager
        self.loss_manager = loss_manager
        self.tensorboard_manager = tensorboard_manager
        self.epoch_prints = epoch_prints

        self.variables = []

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.variables_adder = variables_adder
        self.pre_step = pre_step
        self.post_step = post_step

    def train(self, epochs=100, batch_size=1, evaluation_loss: str = None,
            patience: Number = None, surpress_printing=False) -> None:
        # evaluation_loss is used both for saving the ``best" model
        # every time that loss is minimized and for early termination if that
        # loss doesn't improve after ``patience" epochs
        if patience is None:
            patience = float('inf')

        assert not (math.isfinite(patience) and (evaluation_loss is None))

        # Must be wiped at the start of every training
        file_utils.create_empty_directory(dirs.out_path('renders'))

        # Cutoff is tuple of loss string + patience
        self.system.train()
        optimizer = self.optimizer_factory(self.system.parameters())
        scheduler = None
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            for _ in range(len(self.variables)): scheduler.step()

        dataloader = DataLoader(self.data_manager.train_dataset,
                                batch_size=batch_size, shuffle=True)

        if len(self.variables) == 0:
            with torch.no_grad():
                epoch_vars, losses = self.extract_epoch_variables()
                epoch_vars.update({'time_train': 0.0, 'time_log': 0.0, 'bad_losses': 0, 'epoch': 0,
                                'memory_mb': utils.process_memory()})

                self.finish_step(epoch_vars, losses)

            if evaluation_loss is not None:
                best_cost, no_improvement_n = epoch_vars[evaluation_loss], 0
                self.save_best()
        else:
            if evaluation_loss is not None:
                costs = [epoch_vars[evaluation_loss] for epoch_vars in self.variables]
                best_cost = min(costs)
                no_improvement_n = len(costs) - costs.index(best_cost) - 1

        for epoch in range(len(self.variables), epochs+1):
            start_time = time.time()
            bad_losses = 0
            for batch in dataloader:
                bad_loss = self.training_step(optimizer, batch[0], surpress_printing)
                if bad_loss: bad_losses += 1
            if scheduler is not None: scheduler.step()
            train_end_time = time.time()

            with torch.no_grad():
                epoch_vars, losses = self.extract_epoch_variables()
                log_end_time = time.time()

                epoch_vars.update({'time_train': train_end_time - start_time,
                                'time_log': log_end_time - train_end_time,
                                'bad_losses': bad_losses, 'epoch': epoch,
                                'memory_mb': utils.process_memory()})

                self.finish_step(epoch_vars, losses)

            if evaluation_loss is not None:
                if epoch_vars[evaluation_loss] < best_cost:
                    best_cost, no_improvement_n = epoch_vars[evaluation_loss], 0
                    self.save_best()
                else:
                    no_improvement_n += 1

                if no_improvement_n > patience: break

        logger.info('Finished: saving best losses...')
        self.eval_best()
        self.tensorboard_manager.kill_old_process()

    def training_step(self, optimizer: torch.optim.Optimizer, batch: Tensor,
                            surpress_printing: bool) -> bool:
        bad_loss = False
        save_stdout = sys.stdout
        try:
            optimizer.zero_grad()

            if surpress_printing: sys.stdout = open(dirs.out_path('trash.txt'), 'w')

            loss = self.loss_manager.compute_training_loss(batch)
            loss.backward()

            if surpress_printing: sys.stdout = save_stdout
        except KeyboardInterrupt: sys.exit()
        except Exception as e:
            if surpress_printing: sys.stdout = save_stdout
            logger.error(e)
            traceback.print_exc()
            bad_loss = True

        if not bad_loss:
            if self.pre_step is not None:
                self.pre_step()
            optimizer.step()
            if self.post_step is not None:
                self.post_step()

        return bad_loss

    def finish_step(self, epoch_vars: Dict[str, Number],
                      losses: Dict[str, List[float]]) -> None:
        with torch.no_grad():
            self.variables.append(epoch_vars)
            self.tensorboard_manager.update(self.variables, epoch_vars, losses)
            self.print_epoch(epoch_vars)
            self.save_training()

    def print_epoch(self, epoch_vars: Dict[str, Number]) -> None:
        print_str = f"Epoch: {epoch_vars['epoch']:3}"
        for epoch_print in self.epoch_prints:
            val = round(epoch_vars[epoch_print[0]], 3)
            print_str += f', {epoch_print[1]}: {val:6.3f}'
        logger.info(print_str)

    def extract_epoch_variables(self) -> Dict[str, Number]:
        self.system.eval()
        vars, losses = self.loss_manager.compute_reporting_losses(self.data_manager)
        if self.variables_adder is not None:
            vars.update(self.variables_adder())
        self.system.train()
        return vars, losses

    def save_training(self) -> None:
        torch.save(self.system.state_dict(), dirs.out_path('trainer.pt'))
        with open(dirs.out_path('variables.pickle'), 'wb') as file:
            pickle.dump(self.variables, file)
        with open(dirs.out_path('variables.json'), 'w') as file:
            json.dump(self.variables, file, sort_keys=True, indent=4)

    def load_training(self) -> None:
        self.system.load_state_dict(torch.load(dirs.out_path('trainer.pt')))
        self.system.eval()
        with open(dirs.out_path('variables.pickle'), 'rb') as file:
            self.variables = pickle.load(file)

    def save_best(self) -> None:
        epoch_vars = self.variables[-1]

        file_utils.create_empty_directory(dirs.out_path('best'))
        torch.save(self.system.state_dict(), dirs.out_path('best', 'trainer.pt'))
        with open(dirs.out_path('best', 'epoch_vars.pickle'), 'wb') as file:
            pickle.dump(epoch_vars, file)
        with open(dirs.out_path('best', 'epoch_vars.json'), 'w') as file:
            json.dump(epoch_vars, file, sort_keys=True, indent=4)

    def eval_best(self) -> None:
        self.system.load_state_dict(torch.load(dirs.out_path('best', 'trainer.pt')))
        self.system.eval()
        stats, losses = self.loss_manager.compute_reporting_losses(
                self.data_manager, types=['train', 'valid', 'test'], eval_all=True)

        with open(dirs.out_path('best', 'stats.json'), 'w') as file:
            json.dump(stats, file, sort_keys=True, indent=4)
        with open(dirs.out_path('best', 'losses.json'), 'w') as file:
            json.dump(losses, file, sort_keys=True, indent=4)
