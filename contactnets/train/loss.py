from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pdb  # noqa
import random
from typing import Dict, List, Sized, Tuple, Union, cast

from torch import Tensor
from torch.utils.data import Dataset

from contactnets.system import System
from contactnets.train.data_manager import DataManager
from contactnets.utils import dirs, file_utils, system_io


class LossType(Enum):
    STEPWISE = 1
    TRAJECTORY = 2


class Loss(ABC):
    """Compute losses over an underlying system.

    Attributes:
        names: a name string for each entry returned by compute_loss.
        system: the system to compute the loss over.
        loss_type: whether the loss operates over a trajectory or a single pair of time steps.
        mutates: whether the loss will change the underlying system during calculations.
    """
    names: List[str]
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
        """Compute a loss over the states and controls in the underlying system.

        System may be modified if the loss mutates the system. Can return multiple loss metrics
        for computation efficiency (must modify self.names accordingly). Only the first returned
        loss metric would be used for training.

        Args:
            meta_save_file: where to write expensive intermediate computations to. This is
            useful if after completing training you want to later compute a new loss function
            over already simulated trajectories. The trajectory loss writes model rollouts
            to this file for this purpose.
        """
        pass

    def preprocess_data(self, data: Tensor) -> Tensor:
        """Modify data before it is loaded into the system."""
        return data


@dataclass
class ReportingLoss:
    """A loss function that is not used for gradient descent but for reporting purposes."""
    loss: Loss
    sample_n: int


class LossManager:
    """Manage losses for training and reporting purposes."""
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
        """Sum up training losses over stepwise data.

        If you have multiple training losses, none can mutate the underlying system. All
        training losses must operate on stepwise data.

        Args:
            stepwise_data: batch_n x 2 x (state_n + control_n) * entity_n.
        """
        assert len(self.training_losses) > 0
        system_io.load_system(stepwise_data, self.system)

        def compute_single_loss(loss: Loss) -> Tensor:
            if len(self.training_losses) > 1: assert not loss.mutates
            assert loss.loss_type == LossType.STEPWISE
            return loss.compute_loss()[0]

        losses = [compute_single_loss(loss) for loss in self.training_losses]

        return cast(Tensor, sum(losses))

    def compute_reporting_losses(self, data: DataManager, types=['train', 'valid'],
                                 eval_all=False) -> Tuple[Dict[str, float],
                                                          Dict[str, List[float]]]:
        """Compute reporting losses.

        Uses sample_n in each reporting loss to determine how many times to run the loss
        function.

        Args:
            data: the DataManager containing the data. Must have already loaded.

            types: a sublist of ['train', 'valid', 'test']. What kinds of data to compute
            losses for.

            eval_all: ignore sample_n and run the entire dataset. Used on the final model.

        Returns:
            Two dictionaries whose keys are of the form {data_type}_{loss_name}. For example,
            'train_surrogate' or 'valid_pos_pred'. The first dictionary contains the average
            losses over the sample_n samples for each reporting loss. The second dictionary
            maps to the entire list of sample_n loss samples.
        """
        if eval_all:
            file_utils.create_empty_directory(dirs.out_path('best', 'meta'))

        loss_stats, losses = {}, {}

        def compute(reporting: ReportingLoss, datasets):
            for data_type, dataset in zip(['train', 'valid', 'test'], datasets):
                if data_type in types:
                    reporting_losses = self._reporting_loss(
                        reporting, dataset, data_type, eval_all=eval_all)
                    names = reporting.loss.names
                    assert len(reporting_losses) == len(names)
                    for reporting_loss, name in zip(reporting_losses, names):
                        loss_stats[f'{data_type}_{name}'] = reporting_loss[0]
                        losses[f'{data_type}_{name}'] = reporting_loss[1]

        for reporting in self.reporting_losses:
            if reporting.loss.loss_type == LossType.STEPWISE:
                compute(reporting, [data.train_dataset, data.valid_dataset, data.test_dataset])
            elif reporting.loss.loss_type == LossType.TRAJECTORY:
                compute(reporting, [data.train_runs, data.valid_runs, data.test_runs])
            else:
                raise Exception('LossType not recognized.')

        return loss_stats, losses

    def _reporting_loss(self, reporting: ReportingLoss,
                        dataset: Union[Dataset[Tuple[Tensor]], List[Tensor]],
                        data_type: str, eval_all=False) -> List[Tuple[float, List[float]]]:
        """
        dataset just needs to implement len() and __getitem__().
        The type of dataset should match the type of the loss (i.e. STEPWISE or TRAJECTORY)

        Args:
            reporting: the loss to evaluate.

            dataset: just needs to implement len() and __getitem__(). The type of the dataset
            should match the type of the loss (i.e. STEPWISE or TRAJECTORY).

            data_type: 'train', 'valid', or 'test'. When saving meta data on the last run
            when eval_all = True, indicates which subdirectory to save it under.

            eval_all: ignore sample_n and run the entire dataset. Used on the final model.

        Returns:
            A list whose length is the number of losses computed by the ReportingLoss. For each
            loss, the average loss and the entire list of losses is returned.
        """

        indices = list(range(len(cast(Sized, dataset))))
        random.shuffle(indices)

        if eval_all:
            save_dir = dirs.out_path('best', 'meta', reporting.loss.names[0], data_type)
            file_utils.create_empty_directory(save_dir)
        else:
            indices = indices[:min(reporting.sample_n, len(indices))]

        losses: List[List[float]] = []

        for i in indices:
            data = dataset[i]
            if type(data) is tuple:
                assert len(data) == 1
                data_tensor = data[0].unsqueeze(0)
            else:
                data_tensor = cast(Tensor, data)

            data_tensor = reporting.loss.preprocess_data(data_tensor)
            system_io.load_system(data_tensor, self.system)

            save_file = dirs.out_path('best', 'meta', reporting.loss.names[0],
                                      data_type, f'{i}.pt') if eval_all else None
            new_losses = reporting.loss.compute_loss(meta_save_file=save_file)

            if len(losses) == 0:
                losses = [[] for i in range(len(new_losses))]

            for i, new_loss in enumerate(new_losses):
                losses[i].append(new_loss.item())

        reporting_losses = [(sum(single_losses) / len(indices), single_losses)
                            for single_losses in losses]

        return reporting_losses
