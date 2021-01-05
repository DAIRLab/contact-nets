import os
import pdb  # noqa
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

from contactnets.utils import dirs


class DataManager:
    """Manage loading data.

    Attributes:
        train_runs / valid_runs / test_runs: lists of run tensors of the shape:
        batch_n x step_n x (state_n + control_n) * entity_n.
        Where state_n = configuration_n + velocity_n.

        train_dataset / valid_dataset / test_dataset: lists of TensorDatasets where each element
        is a singleton tuple tensor of the shape:
        2 x (state_n + control_n) * entity_n.
        This is a pair of previous and next time step states.

        device: the device to move loaded tensors to.
    """
    train_runs: List[Tensor]
    valid_runs: List[Tensor]
    test_runs: List[Tensor]

    # Tuples should always have one element
    train_dataset: Dataset[Tuple[Tensor, ...]]
    valid_dataset: Dataset[Tuple[Tensor, ...]]
    test_dataset: Dataset[Tuple[Tensor, ...]]

    device: torch.device

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def load(self, noise=0.0) -> None:
        """Load the dataset into the object attributes.

        Args:
            noise: how much Gaussian noise to add.
        """
        def process_run(data: Tensor) -> Tensor:
            """ Turn time sequenced data into a batch of paired time steps.

            Args:
                data: 1 x step_n x (state_n + control_n) * entity_n.

            Returns:
                step_n - 1 x 2 x (state_n + control_n) * entity_n
            """
            assert data.shape[0] == 1

            n = data.shape[1]
            batch = torch.zeros((n - 1, 2, data.shape[2]))

            for i in range(n - 1):
                batch[i, :, :] = torch.cat((data[0, i, :].unsqueeze(0),
                                            data[0, i + 1, :].unsqueeze(0)))
            return batch.to(self.device)

        def load_runs(subdir: str) -> List[Tensor]:
            datas = []

            try:
                for file_name in os.listdir(dirs.out_path('data', subdir)):
                    file_path = dirs.out_path('data', subdir, file_name)
                    data = torch.load(file_path).to(self.device)
                    data = data + torch.randn(data.shape) * noise
                    datas.append(data)
            except Exception as err:
                print(f'Could not load data for {subdir}, did you run experiment/split.py?')
                print(err)

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
