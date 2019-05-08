"""This generates a figure of a few random MNIST digits of particular
labels and then also a small animation in which they are stacked
vertically
"""

import shared.setup_torch # pylint: disable=unused-import
import shared.filetools
import torch
import numpy as np
import mnist.pwl
import matplotlib as mpl
import matplotlib.animation as mplanim
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from os.path import join as pjoin

SAVEDIR = shared.filetools.savepath()

CMAP = np.zeros((258, 4))
CMAP[2:, 0] = np.linspace(1, 0, 256)
CMAP[2:, 1] = CMAP[2:, 0]
CMAP[2:, 2] = CMAP[2:, 0]
CMAP[2:, 3] = 1
CMAP[1] = CMAP[2]
CMAP = mpl.colors.ListedColormap(CMAP)
def plot_img(ax, points, noticks=False):
    """Plots the image represented by the given flattened torch tensor

    Args:
        ax (matplotlib.axes.Axes): the axes to plot onto
        points (torch.tensor[28*28]): the flattened data

    Returns:
        result from imshow
    """
    ticklocs = tuple(range(0, 28, 9))
    tickvals = tuple(str(val) for val in range(1, 29, 9))

    reshaped = (points + 2).reshape(28, 28).numpy()
    res = ax.imshow(reshaped, cmap=CMAP, vmin=1)
    if not noticks:
        ax.set_xticks(ticklocs)
        ax.set_xticklabels(tickvals)
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(tickvals)
    return res

def plot_img_stacked(ax, points):
    """Plots the image represented by the given flattened torch tensor
    as vertical columns

    Args:
        ax (matplotlib.axes.Axes): the axes to plot onto
        points (torch.tensor[28*28]): the flattened data

    Returns:
        result from imshow
    """
    reshaped = (points + 2).reshape(28, 28).transpose(0, 1).flatten().reshape(1, -1).transpose(0, 1).numpy()
    res = ax.imshow(reshaped, cmap=CMAP, vmin=1, aspect=1/30)
    ax.set_xticks((0,))
    ax.set_xticklabels(('1',))
    return res


def plot_img_spread(ax, points, spacing=4):
    """Plots the image represented by the given flattened torch tensor, except adding
    one column of transparent pixels between each column

    Args:
        ax (matplotlib.axes.Axes): the axes to plot onto
        points (torch.tensor[28*28]): the flattened image to plot

    Returns:
        result from imshow
    """

    reshaped = points.reshape(28, 28).numpy()
    spaced = np.zeros((28, 28 + 27*(spacing-1)), dtype=reshaped.dtype)
    spaced[:] = -2

    spaced[:, np.arange(0, 28*spacing, spacing)] = reshaped
    spaced += 1
    res = ax.imshow(spaced, cmap=CMAP)

    xticklocs = np.arange(0, 28*spacing, spacing*9)
    xtickvals = tuple(str(val) for val in range(1, 29, 9))
    yticklocs = np.arange(0, 28, 9)
    ytickvals = xtickvals
    ax.set_xticks(xticklocs)
    ax.set_xticklabels(xtickvals)
    ax.set_yticks(yticklocs)
    ax.set_yticklabels(ytickvals)
    return res


def main():
    """Main runner"""
    pwl = mnist.pwl.MNISTData.load_test().to_pwl() # test smaller -> faster to load

    points = torch.zeros((4, 28*28), dtype=torch.double)
    labels = torch.zeros((4,), dtype=torch.int)

    pwl.fill(points, labels)


    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for i, ax in enumerate(np.array(axs).flatten()):
        plot_img(ax, points[i])

    os.makedirs(SAVEDIR, exist_ok=True)
    fig.savefig(pjoin(SAVEDIR, 'fig1.png'), transparent=True, dpi=200)
    plt.close(fig)
    fig, axs = None, None

    fig, ax = plt.subplots()
    plot_img(ax, points[0])
    fig.savefig(pjoin(SAVEDIR, 'fig2.png'), transparent=True, dpi=200)
    plt.close(fig)
    fig, ax = None, None

    for i in range(3, 6):
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = plt.axes()
        plot_img_spread(ax, points[0], spacing=i-1)
        ax.set_aspect(1)
        fig.savefig(pjoin(SAVEDIR, f'fig{i}.png'), transparent=True, dpi=200)
        plt.close(fig)
        fig, ax = None, None

    fig, ax = plt.subplots()
    plot_img_stacked(ax, points[0])
    fig.savefig(pjoin(SAVEDIR, f'fig6.png'), transparent=True, dpi=200)
    plt.close(fig)
    fig, ax = None, None


def crop(imgpath):
    img = Image.open(imgpath)
    img.load()
    image_data = np.asarray(img)
    image_data_bw = image_data[:, :, 3]
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    crop_box = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    image_data_new = image_data[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1 , :]

    new_image = Image.fromarray(image_data_new)
    new_image.save(imgpath)

def plot_randoms_stacked():
    for i in range(7, 15):
        points = torch.from_numpy(np.random.uniform(2, 258, size=(28, 28)))
        fig, ax = plt.subplots()
        plot_img_stacked(ax, points)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(pjoin(SAVEDIR, f'fig{i}.png'), dpi=200, transparent=True)
        crop(pjoin(SAVEDIR, f'fig{i}.png'))


def plot_one_of_each():
    pwl = mnist.pwl.MNISTData.load_test().to_pwl() # test smaller -> faster to load

    points = torch.zeros((10, 28*28), dtype=torch.double)
    labels = torch.zeros((10,), dtype=torch.int)

    pts = torch.zeros((1, 28*28), dtype=torch.double)
    lbls = torch.zeros(1, dtype=torch.int)
    for i in range(10):
        pwl.restrict_to(labels={i}).fill(pts, lbls)
        points[i] = pts
        labels[i] = lbls[0]


    fig, axs = plt.subplots(1, 10)

    for i, ax in enumerate(np.array(axs).flatten()):
        ax.set_xticks([])
        ax.set_yticks([])
        plot_img(ax, points[i], True)

    os.makedirs(SAVEDIR, exist_ok=True)
    fig.savefig(pjoin(SAVEDIR, 'fig15.png'), transparent=True, dpi=200)
    plt.close(fig)
    fig, axs = None, None

    crop(pjoin(SAVEDIR, 'fig15.png'))


if __name__ == '__main__':
    plot_one_of_each()