from tensorboardX import SummaryWriter

import torch
from torch import Tensor
from torch.nn import Sequential

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib.ticker import PercentFormatter
import mplcursors
import threading

import os, subprocess
import numpy as np
from numbers import Number

import socket

from typing import *
from dataclasses import dataclass

from contactnets.utils import utils, dirs, file_utils
from contactnets.jacnet import Series

import pdb

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

@dataclass
class TensorboardPlot:
    """
    Convenient struct to encapsulate simple line plots.
    ax1_vars and ax2_vars specify variables to be plotted against each axis.
    """
    name:     str
    ax1_vars: List[str]
    ax2_vars: List[str]
    log_axis: bool # Log scale y axis

@dataclass
class TensorboardSeriesHistogram:
    """
    Convenient struct for visualizing data passing through a jacnet Series module (or torch Sequential)
    data is a tensor of the form num_data_points x data_point_size.
    data_point_size should be the size of the expected network input.
    """
    name: str
    series: Union[Series, Sequential]
    data: Tensor

class TensorboardManager:
    writer: SummaryWriter
    plots: List[TensorboardPlot]
    histograms: List[TensorboardSeriesHistogram]

    callback: Callable[[SummaryWriter, Dict[str, Number]], None]

    def __init__(self, callback=None):
        self.callback = callback

        self.plots = []
        self.histograms = []

    def create_writer(self, resume=False) -> None:
        if not resume:
            file_utils.clear_directory(dirs.out_path('tensorboard'))
        self.writer = SummaryWriter(dirs.out_path('tensorboard'))

    def kill_old_process(self) -> None:
        pass
        # command = 'kill -9 $(lsof -t -i:6006) > /dev/null 2>&1'
        # os.system(command)

    def launch(self) -> None:
        self.kill_old_process()

        # Use threading so tensorboard is automatically closed on process end
        command = 'tensorboard --samples_per_plugin images=0 --bind_all --port 6006 --logdir {} > /dev/null --window_title {} 2>&1'.format(dirs.out_path('tensorboard'), socket.gethostname())
        t = threading.Thread(target=os.system, args=(command,))
        t.start()

    def update(self, history_vars: List[Dict[str, Number]],
                     epoch_vars:   Dict[str, Number],
                     losses:       Dict[str, List[Number]]) -> None:
        history_vars = utils.list_dict_swap(history_vars)

        self.__write_scalars(epoch_vars)

        for plot in self.plots:
            self.__write_plot(plot, history_vars)

        for hist in self.histograms:
            self.__write_histogram(hist, epoch_vars['epoch'])

        if self.callback:
            self.callback(self.writer, epoch_vars)

        # TODO: should just have to flush--for some reason, need to close the writer??
        self.writer.close()

    def __write_scalars(self, epoch_vars: Dict[str, Number]) -> None:
        for field in epoch_vars.keys():
            self.writer.add_scalar(field, torch.tensor(epoch_vars[field]), epoch_vars['epoch'])

    def __write_plot(self, plot: TensorboardPlot,
                                 history_vars: Dict[str, List[Number]]) -> None:
        fig = plt.figure(figsize=(8,5))
        ax1 = plt.subplot(111)

        if len(plot.ax2_vars) == 0:
            active_vars = list(filter(lambda var: max(history_vars[var]) > 1e-7, plot.ax1_vars))

            if len(active_vars) <= 10:
                colors = plt.cm.tab10.colors
            else:
                colors = plt.cm.tab20.colors

            train_i, valid_i = 0, 0
            for var in active_vars:
                # Remove small numbers to prevent plot scale messup
                vars_zeroed = [0 if x < 1e-7 else x for x in history_vars[var]]
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

    def __write_histogram(self, hist: TensorboardSeriesHistogram, epoch: int) -> None:
        x = hist.data

        if isinstance(hist.series, Series):
            modules = hist.series.jac_modules
        else:
            modules = hist.series

        for i, module in enumerate(modules):
            if isinstance(hist.series, Series):
                module = module.operation

            self.writer.add_histogram(f'{hist.name}_layer{i}/a_pre',
                x.cpu().detach().numpy(), epoch)

            x = module(x)

            self.writer.add_histogram(f'{hist.name}_layer{i}/b_post',
                x.cpu().detach().numpy(), epoch)

            if isinstance(module, torch.nn.Linear):
                self.writer.add_histogram(f'{hist.name}_layer{i}/c_weights',
                        module.weight.cpu().detach().numpy(), epoch)

                if module.bias is not None:
                    self.writer.add_histogram(f'{hist.name}_layer{i}/d_biases',
                            module.bias.cpu().detach().numpy(), epoch)
