"""This module is to test plotting things in 3d interactively
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from gaussian_spheres.pwl import GaussianSpheresPWLP

NUM_POINTS = 1000
NUM_CLUSTERS = 50
NUM_LABELS = 2

def main():
    """Main runner"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = torch.zeros((NUM_POINTS, 3), dtype=torch.double)
    labels = torch.zeros(NUM_POINTS, dtype=torch.long)
    pwl = GaussianSpheresPWLP.create(NUM_POINTS, 3, NUM_LABELS, 2, NUM_CLUSTERS, 0.04, 0, 0.08)
    pwl.fill(points, labels)

    ax.scatter(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), s=1, c=labels.numpy(), label='3d clusters')
    ax.legend()


    fig.canvas.mpl_connect('close_event', exit)

    while True:
        for angle in range(360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(0.01)


if __name__ == '__main__':
    main()
