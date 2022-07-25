import torch
import numpy as np
import matplotlib.pyplot as plt


def map_device(tensors, net):
    """
    Maps a list of tensors to the device used by the network net
    :param tensors: list of tensors
    :param net: nn.Module
    :return: list of tensors
    """
    if net.wi.device != torch.device('cpu'):
        new_tensors = []
        for tensor in tensors:
            new_tensors.append(tensor.to(device=net.wi.device))
        return new_tensors
    else:
        return tensors


def center_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.set(xticks=[], yticks=[])



def radial_distribution_plot(angles, N=80, bottom=.1, cmap_scale=0.05):
    """
    Plot a radial histogram of angles
    :param angles: a series of angles
    :param N: num bins
    :param bottom: radius of base circle
    :param cmap_scale: to adjust the colormap
    :return:
    """
    angles = angles % (2 * np.pi)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = [np.mean(np.logical_and(angles > theta[i], angles < theta[i+1])) for i in range(len(theta)-1)]
    radii.append(np.mean(angles > theta[-1]))
    width = (2*np.pi) / N
    offset = np.pi / N
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta + offset, radii, width=width, bottom=bottom)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / cmap_scale))
        bar.set_alpha(0.8)
    plt.yticks([])