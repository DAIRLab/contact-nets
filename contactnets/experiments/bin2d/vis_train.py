from tensorboardX import SummaryWriter

import torch

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib.ticker import PercentFormatter
import mplcursors

import numpy as np

from contactnets.utils import dirs, utils

import pdb

font_small = 18
font_medium = 20
font_big = 22

plt.switch_backend('TkAgg')

plt.rc('font', size=font_small)
plt.rc('axes', titlesize=font_small)
plt.rc('axes', labelsize=font_medium)
plt.rc('xtick', labelsize=font_small)
plt.rc('ytick', labelsize=font_small)
plt.rc('legend', fontsize=font_small)
plt.rc('figure', titlesize=font_big)

writer = None

def init(resume=False):
    global writer
    if not resume:
        utils.clear_directory(dirs.out_path('tensorboard'))

    writer = SummaryWriter(dirs.out_path('tensorboard'))

def plot_loss(variables, plot_validation=True, log_axes=True):
    fig = plt.figure(figsize=(15,10))

    ax1 = plt.subplot(111)

    ax2 = ax1.twinx()

    surrogate_color = '#1f77b4'
    prediction_color = '#ff7f0e'
    
    ax1.plot(variables["training_loss_surrogate"],
             color=surrogate_color, linewidth=4, alpha=1)

    ax2.plot(variables["training_loss"],
             color=prediction_color, linewidth=4, alpha=1)
    
    if plot_validation:
        ax1.plot(variables["validation_loss_surrogate"],
                 color=surrogate_color, linewidth=4, alpha=0.5)
        ax2.plot(variables["validation_loss"],
                 color=prediction_color, linewidth=4, alpha=0.5)

        ax1.legend(['Train', 'Validation'])
    
    ax1.tick_params(axis='y', labelcolor=surrogate_color)
    ax1.set_ylabel('Surrogate loss', color=surrogate_color)
    ax1.set_xlabel('Epochs')

    ax2.tick_params(axis='y', labelcolor=prediction_color)
    ax2.set_ylabel('L2 state prediction error', color=prediction_color)
    
    if log_axes:
        ax1.set_yscale('log')
        ax2.set_yscale('log')

    ax2.set_xlabel('Epochs')
    
    plt.tight_layout()
    
    writer.add_figure('a_loss', fig)

def log_l2_loss_histograph(epoch_vars):
    fig = plt.figure(figsize=(10,10))
    
    ax = plt.subplot(111)

    kwargs = dict(alpha=0.5, density=False, stacked=False)

    tl = epoch_vars['training_losses'] 
    vl = epoch_vars['validation_losses'] 
    
    bins = np.histogram(np.hstack((tl, vl)), bins=50)[1]
    plt.hist(tl, bins, **kwargs, color='g', label='Training')
    plt.hist(vl, bins, **kwargs, color='b', label='Validation')

    ax.set_title('L2 Loss Distribution')
    ax.set_xlabel('L2 Loss')
    ax.set_ylabel('Count')

    plt.legend()

    writer.add_figure('c_l2_loss_distribution', fig)

def log_surrogate_loss_histograph(epoch_vars):
    fig = plt.figure(figsize=(10,10))
    
    ax = plt.subplot(111)

    kwargs = dict(alpha=0.5, density=False, stacked=False)
    
    tl = epoch_vars['training_losses_surrogate']
    vl = epoch_vars['validation_losses_surrogate'] 
    
    bins = np.histogram(np.hstack((tl, vl)), bins=50)[1]
    plt.hist(tl, bins, **kwargs, color='g', label='Training')
    plt.hist(vl, bins, **kwargs, color='b', label='Validation')

    ax.set_title('Surrogate Loss Distribution')
    ax.set_xlabel('Surrogate Loss')
    ax.set_ylabel('Count')

    plt.legend()

    writer.add_figure('d_surrogate_loss_distribution', fig)

def log_model_normal_layers(system, dataset, epoch):
    x = torch.zeros((len(dataset), 3))
    for i, entry in enumerate(dataset):
        x[i, :] = dataset[i][0][0][0:3]
    
    for i, module in enumerate(system.interactions[0].dynamics.phi_net.root.operation.jac_modules):
        module = module.operation

        writer.add_histogram('layer{}/a_pre'.format(i),
            x.detach().numpy(), epoch)

        x = module(x)

        writer.add_histogram('layer{}/b_post'.format(i),
            x.detach().numpy(), epoch)

        if isinstance(module, torch.nn.Linear):
            writer.add_histogram('layer{}/c_weights'.format(i),
                module.weight.detach().numpy(), epoch)
            
            if module.bias is not None:
                writer.add_histogram('layer{}/d_biases'.format(i),
                    module.bias.detach().numpy(), epoch)
        
def log_model_normal(system):
    fig = plt.figure(figsize=(10,10))
    
    ax = plt.subplot(111)

    plt.grid()

    Jn = system.interactions[0].dynamics.phi_net.root.operation.jac_modules[1].operation.weight
    #Jn = model.phi_net.root.operation.jac_modules[1].operation.weight 
    #Jn = model.phi_net.root.operation.jac_modules[1].operation.jac_modules[0].operation.weight 
    
    corners = Jn[:, 2:4]
    corners = corners.detach().cpu().numpy()

    ax.fill(corners[:, 0], corners[:, 1], color=[1.0, 0, 0])

    normals = Jn[:, 0:2]
    normals = normals.detach().cpu().numpy()

    for i, corner in enumerate(corners):
        ax.arrow(corners[i, 0], corners[i, 1], normals[i, 0], normals[i, 1], width=0.01)

    ax.set(xlim=(-3, 3), ylim=(-3, 3))

    writer.add_figure('e_normal', fig)

def log_model_tangent(system):
    fig = plt.figure(figsize=(10,10))
    
    ax = plt.subplot(111)

    plt.grid()

    Jt_tilde = system.interactions[0].dynamics.tangent_net.root.operation.jac_modules[1].operation.weight

    corners = Jt_tilde[:, 2:4]
    corners = corners.detach().cpu().numpy()

    ax.fill(corners[:, 0], corners[:, 1], color=[0.0, 0, 1.0])

    tangents = Jt_tilde[:, 0:2]
    tangents = tangents.detach().cpu().numpy()

    for i, corner in enumerate(corners):
        ax.arrow(corners[i, 0], corners[i, 1], tangents[i, 0], tangents[i, 1], width=0.01)

    ax.set(xlim=(-3, 3), ylim=(-3, 3))

    writer.add_figure('e_tangent', fig)
