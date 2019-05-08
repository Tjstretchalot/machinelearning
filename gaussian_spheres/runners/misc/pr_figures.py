"""These plots are helpful for talking about helpful dimensionality via participation
ratio
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
from os.path import join as pjoin
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import math

#   https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

SAVEDIR = shared.filetools.savepath()

def plot(points, lines, scatterargs, lineargs, fontargs):
    """Plots the specified points in a pretty way without axes or anything

    Args:
        points (np.ndarray): the points to plot
        lines (list[point1, point2, labelpt]):
            each point is a point the line should go through except labelpt is
            where we put the label

    Returns:
        figure, axes
    """

    if 's' not in scatterargs:
        scatterargs['s'] = 4

    if 'c' not in scatterargs:
        scatterargs['c'] = 'green'

    if 'alpha' not in scatterargs:
        scatterargs['alpha'] = 0.4

    if 'linewidth' not in lineargs:
        lineargs['linewidth'] = 2

    if 'color' not in lineargs:
        lineargs['color'] = 'black'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_aspect('equal')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **scatterargs)

    for i, line in enumerate(lines):
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]],
                **lineargs)

        ax.text(line[2][0], line[2][1], line[2][2], f'$\\lambda_{i}$', **fontargs)

    fig.tight_layout()
    return fig, ax

def fig1(scatterargs, lineargs, fontargs, savefigkwargs):
    """entry"""

    npoints = 300
    points = np.random.randn(npoints, 3)

    #norm = np.linalg.norm(points, axis=1)
    #points *= np.power(np.random.uniform(0, radius), 1/3) / norm[:, np.newaxis]

    os.makedirs(SAVEDIR, exist_ok=True)

    fig, ax = plot(points, [
                    ((-3, 0, 0), (3, 0, 0), (-3, 0, 0.5)),
                    ((0, -3, 0), (0, 3, 0), (-0.3, -3.5, 0.5)),
                    ((0, 0, -3), (0, 0, 3), (0, 0.5, -3))
              ], scatterargs, lineargs, fontargs)
    fig.savefig(pjoin(SAVEDIR, 'fig1.png'), **savefigkwargs)
    plt.close(fig)
    fig, ax = None, None

def fig2(scatterargs, lineargs, fontargs, savefigkwargs):
    npoints = 300
    points = np.random.randn(npoints, 3)
    points[:, 0] *= 3
    points[:, 1:] *= 0.3

    #norm = np.linalg.norm(points, axis=1)
    #points *= np.power(np.random.uniform(0, radius), 1/3) / norm[:, np.newaxis]

    os.makedirs(SAVEDIR, exist_ok=True)

    fig, ax = plot(points, [
                    ((-9, 0, 0), (9, 0, 0), (-9, 0, 0.1)),
                    ((0, -1, 0), (0, 1, 0), (0, -1, -0.3)),
                    ((0, 0, -1), (0, 0, 1), (0, 0.2, -1))
              ], scatterargs, lineargs, fontargs)
    fig.savefig(pjoin(SAVEDIR, 'fig2.png'), **savefigkwargs)
    plt.close(fig)
    fig, ax = None, None


def fig3(scatterargs, lineargs, fontargs, savefigkwargs):
    npoints = 300
    points = np.random.randn(npoints, 3)

    vecs = np.array([
            ((-3, 0, 0), (3, 0, 0), (-3.5, 0, -1)),
            ((0, -3, 0), (0, 3, 0), (0, -3.2, 0)),
            ((0, 0, -3), (0, 0, 3), (0, 0, 3.5))
        ])

    rescalematrix = np.array([
        [3.0, 0, 0],
        [0, 3.0, 0],
        [0, 0, 1/3]
    ])

    rotatematrix = rotation_matrix(np.array([0, 1, 1], dtype='float64'), 30)

    points = points @ rescalematrix
    points = points @ rotatematrix
    vecs = vecs @ rescalematrix
    vecs = vecs @ rotatematrix


    #norm = np.linalg.norm(points, axis=1)
    #points *= np.power(np.random.uniform(0, radius), 1/3) / norm[:, np.newaxis]

    os.makedirs(SAVEDIR, exist_ok=True)

    fig, ax = plot(points, vecs, scatterargs, lineargs, fontargs)
    fig.savefig(pjoin(SAVEDIR, 'fig3.png'), **savefigkwargs)
    plt.close(fig)
    fig, ax = None, None

def fig4(scatterargs, lineargs, fontargs, savefigkwargs):
    npoints = 300
    points = np.random.randn(npoints*2, 3)
    points[:, :] *= 0.3
    points[:npoints, 0] -= 6
    points[npoints:, 0] += 6

    #norm = np.linalg.norm(points, axis=1)
    #points *= np.power(np.random.uniform(0, radius), 1/3) / norm[:, np.newaxis]

    os.makedirs(SAVEDIR, exist_ok=True)

    fig, ax = plot(points, [
                    ((-9, 0, 0), (9, 0, 0), (-9, 0, 0.3)),
                    ((0, -1, 0), (0, 1, 0), (0, -1.2, -0.5)),
                    ((0, 0, -1), (0, 0, 1), (0, 0.1, -1.5))
              ], scatterargs, lineargs, fontargs)
    fig.savefig(pjoin(SAVEDIR, 'fig4.png'), **savefigkwargs)
    plt.close(fig)
    fig, ax = None, None



if __name__ == '__main__':
    #fig1({'c': '#cccccc'}, {'color': 'white'}, {'color': 'white'}, {'transparent': True, 'dpi': 400})
    #fig2({'c': '#cccccc'}, {'color': 'white'}, {'color': 'white'}, {'transparent': True, 'dpi': 400})
    #fig3({'c': '#cccccc'}, {'color': 'white'}, {'color': 'white'}, {'transparent': True, 'dpi': 400})
    fig4({'c': '#cccccc'}, {'color': 'white'}, {'color': 'white'}, {'transparent': True, 'dpi': 400})