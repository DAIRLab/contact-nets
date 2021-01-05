from dataclasses import dataclass
import os
import pdb  # noqa
import signal
import socket
import threading
from typing import Callable, Dict, List, cast

import matplotlib.pyplot as plt
import psutil
from tensorboardX import SummaryWriter
import torch
from torch import Tensor
from torch.nn import Sequential

from contactnets.utils import dirs, file_utils, utils

font_small = 18
font_medium = 20
font_big = 22

plt.switch_backend('Agg')

plt.rc('font', size=font_small)
plt.rc('axes', titlesize=font_small)
plt.rc('axes', labelsize=font_medium)
plt.rc('xtick', labelsize=font_small)
plt.rc('ytick', labelsize=font_small)
plt.rc('legend', fontsize=font_small)
plt.rc('figure', titlesize=font_big)

# Turn off warnings about too many figures open
plt.rcParams.update({'figure.max_open_warning': 0})


@dataclass
class TensorboardPlot:
    """Convenient struct to encapsulate simple line plots.

    Attributes:
        name: what to name the plot.
        ax1_vars / ax2_vars: which variables to plot against each axis.
        log_axis: make the y axis log scale
    """
    name:     str
    ax1_vars: List[str]
    ax2_vars: List[str]
    log_axis: bool


@dataclass
class TensorboardSequentialHistogram:
    """Convenient struct for visualizing data passing through a torch Sequential.

    Attributes:
        name: what to name the histogram.
        sequential: the sequential module to pass through the data.
        data: batch_n x input_size.
    """
    name: str
    sequential: Sequential
    data: Tensor


class TensorboardManager:
    """Manage logging of the training process.

    Attributes:
        writer: the TensorboardX writer for logging to the tensorboard files.

        plots / histograms: the structs to log each epoch.

        callback: an optional callback to log additional stuff. Takes the writer
        and a dictionary of the variables computed for the epoch as arguments.
    """
    writer: SummaryWriter
    plots: List[TensorboardPlot]
    histograms: List[TensorboardSequentialHistogram]

    callback: Callable[[SummaryWriter, Dict[str, float]], None]

    def __init__(self, callback=None):
        self.callback = callback

        self.plots = []
        self.histograms = []

    def create_writer(self, resume=False) -> None:
        if not resume:
            file_utils.clear_directory(dirs.out_path('tensorboard'))
        self.writer = SummaryWriter(dirs.out_path('tensorboard'))

    def kill_old_process(self) -> None:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            os.kill(child.pid, signal.SIGTERM)


    def launch(self) -> None:
        self.kill_old_process()

        # Use threading so tensorboard is automatically closed on process end
        command = 'tensorboard --samples_per_plugin images=0 --bind_all --port 6006 '\
                  '--logdir {} > /dev/null --window_title {} 2>&1'\
                  .format(dirs.out_path('tensorboard'), socket.gethostname())
        t = threading.Thread(target=os.system, args=(command,))
        t.start()

        print('Launching tensorboard on http://localhost:6006')

    def update(self, history_vars: List[Dict[str, float]], epoch_vars: Dict[str, float],
               losses: Dict[str, List[float]]) -> None:
        """Performs all logging to the tensorboard files."""
        history_vars_swap = utils.list_dict_swap(history_vars)

        self.__write_scalars(epoch_vars)

        for plot in self.plots:
            self.__write_plot(plot, history_vars_swap)

        for hist in self.histograms:
            self.__write_histogram(hist, cast(int, epoch_vars['epoch']))

        if self.callback:
            self.callback(self.writer, epoch_vars)  # type: ignore

        # TODO: should just have to flush--for some reason, need to close the writer??
        self.writer.close()

    def __write_scalars(self, epoch_vars: Dict[str, float]) -> None:
        for field in epoch_vars.keys():
            self.writer.add_scalar(field, torch.tensor(epoch_vars[field]), epoch_vars['epoch'])

    def __write_plot(self, plot: TensorboardPlot,
                     history_vars: Dict[str, List[float]]) -> None:
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot(111)

        if len(plot.ax2_vars) == 0:
            def is_active(var): return max(history_vars[var]) > 1e-7
            active_vars = list(filter(is_active, plot.ax1_vars))

            if len(active_vars) <= 10:
                colors = plt.cm.tab10.colors
            else:
                colors = plt.cm.tab20.colors

            train_i, valid_i = 0, 0
            for var in active_vars:
                # Remove small numbers to prevent plot scale messup
                vars_zeroed = [0 if float(x) < 1e-7 else x for x in history_vars[var]]
                if 'train' in var:
                    ax1.plot(vars_zeroed, linestyle='-',
                             color=colors[train_i], linewidth=4, alpha=0.8, label=var)
                    train_i += 1
                else:
                    ax1.plot(vars_zeroed, linestyle='--',
                             color=colors[valid_i], linewidth=4, alpha=0.8, label=var)
                    valid_i += 1

            ax1.legend(active_vars, loc='upper right', fontsize=8)
            ax1.tick_params(axis='y')
            ax1.set_xlabel('Epochs')

            if plot.log_axis: ax1.set_yscale('log')

            plt.tight_layout()
            self.writer.add_figure(plot.name, fig)
        else:
            ax2 = ax1.twinx()

            ax1_color = 'blue'
            ax2_color = 'orange'

            linestyles = ['-', '--', ':', '-.']
            for var, linestyle in zip(plot.ax1_vars, linestyles):
                ax1.plot(history_vars[var], linestyle=linestyle,
                         color=ax1_color, linewidth=4, alpha=0.9, label=var)

            for var, linestyle in zip(plot.ax2_vars, linestyles):
                ax2.plot(history_vars[var], linestyle=linestyle,
                         color=ax2_color, linewidth=4, alpha=0.9, label=var)

            ax1.legend(plot.ax1_vars)
            ax1.tick_params(axis='y', labelcolor=ax1_color)
            ax1.set_xlabel('Epochs')

            ax2.legend(plot.ax2_vars)
            ax2.tick_params(axis='y', labelcolor=ax2_color)
            ax2.set_xlabel('Epochs')

            if plot.log_axis:
                ax1.set_yscale('log')
                ax2.set_yscale('log')

            plt.tight_layout()
            self.writer.add_figure(plot.name, fig)

        plt.close(fig)

    def __write_histogram(self, hist: TensorboardSequentialHistogram, epoch: int) -> None:
        x = hist.data

        modules = hist.sequential
        def to_numpy(x): return x.cpu().detach().numpy()

        for i, module in enumerate(modules):
            self.writer.add_histogram(f'{hist.name}_layer{i}/a_pre', to_numpy(x), epoch)

            x = module(x)

            self.writer.add_histogram(f'{hist.name}_layer{i}/b_post', to_numpy(x), epoch)

            if isinstance(module, torch.nn.Linear):
                self.writer.add_histogram(f'{hist.name}_layer{i}/c_weights',
                                          to_numpy(module.weight), epoch)

                if module.bias is not None:
                    self.writer.add_histogram(f'{hist.name}_layer{i}/d_biases',
                                              to_numpy(module.bias), epoch)
