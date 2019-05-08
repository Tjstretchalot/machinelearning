"""Some miscellaneous figures for compression theory"""

import shared.setup_torch # pylint: disable=unused-import
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import shared.filetools

SAVEDIR = shared.filetools.savepath()

def fig1():
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot([-8, 8], [0, 0], linewidth=2, color='black', markersize=0)
    ax.plot([0, 0], [-4, 4], linewidth=2, color='black', markersize=0)

    cluster1 = np.random.randn(50, 2)
    cluster2 = np.random.randn(50, 2)

    cluster1[:, 0] -= 5
    cluster2[:, 0] += 5

    combined = np.vstack((cluster1, cluster2))

    ax.scatter(combined[:, 0], combined[:, 1], s=3, c='#333333')

    fig.savefig(pjoin(SAVEDIR, 'fig1.png'), transparent=True, dpi=200)

def fig2():
    # FIG1
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot([-8, 8], [0, 0], linewidth=2, color='black', markersize=0)
    ax.plot([0, 0], [-4, 4], linewidth=2, color='black', markersize=0)

    cluster1 = np.random.randn(50, 2)
    cluster2 = np.random.randn(50, 2)

    cluster1[:, 0] *= 0.5
    cluster2[:, 0] *= 0.5
    cluster1[:, 1] *= 7
    cluster2[:, 1] *= 7

    cluster1[:, 0] -= 5
    cluster2[:, 0] += 5

    ax.scatter(cluster1[:, 0], cluster1[:, 1], s=3, c='#00ffff')
    ax.scatter(cluster2[:, 0], cluster2[:, 1], s=3, c='#ffff00')

    ax.scatter([-5, 5], [0, 0], s=12, c='#ff0000')

    mydir = pjoin(SAVEDIR, 'fig2')
    os.makedirs(mydir, exist_ok=True)
    fig.savefig(pjoin(mydir, 'fig.png'), transparent=True, dpi=200)

    # PROJ
    proj_matrix = np.array([[1.0], [0]]) @ np.array([[1.0, 0]])

    cluster1_proj = cluster1 @ proj_matrix #(proj_matrix @ cluster1.T).T
    cluster2_proj = cluster2 @ proj_matrix #(proj_matrix @ cluster2.T).T

    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



    ax.plot([-8, 8], [0, 0], linewidth=2, color='black', markersize=0, alpha=0.4)
    ax.plot([0, 0], [-4, 4], linewidth=2, color='black', markersize=0, alpha=0.4)

    ax.scatter(cluster1_proj[:, 0], cluster1_proj[:, 1], s=3, c='#00ffff')
    ax.scatter(cluster2_proj[:, 0], cluster2_proj[:, 1], s=3, c='#ffff00')
    ax.scatter([-5, 5], [0, 0], s=12, c='#ff0000')

    fig.savefig(pjoin(mydir, 'proj.png'), transparent=True, dpi=200)

    # FIG2
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



    ax.plot([-8, 8], [2, -2], linewidth=2, color='black', markersize=0, alpha=0.4)

    ax.scatter(cluster1[:, 0], cluster1[:, 1], s=3, c='#00ffff')
    ax.scatter(cluster2[:, 0], cluster2[:, 1], s=3, c='#ffff00')
    #ax.scatter([-5, 5], [0, 0], s=12, c='#ff0000')

    fig.savefig(pjoin(mydir, 'fig2.png'), transparent=True, dpi=200)

    # FIG2_PROJ
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    vec = np.array([16, -4.0])
    uvec = vec / np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
    proj_matrix = (np.array([[16.0], [-4.0]]) @ np.array([[16.0, -4.0]])) / (16*16 + 4*4)

    cluster1_proj = np.zeros_like(cluster1)
    for i, row in enumerate(cluster1):
        dotprod = row[0] * uvec[0] + row[1] * uvec[1]
        cluster1_proj[i] = dotprod * uvec

    cluster2_proj = cluster2 @ proj_matrix
    targets = (np.array([ [-5, 0], [5, 0] ])) @ proj_matrix

    ax.plot([-8, 8], [2, -2], linewidth=2, color='black', markersize=0, alpha=0.4)

    ax.scatter(cluster1_proj[:, 0], cluster1_proj[:, 1], s=3, c='#00ffff')
    ax.scatter(cluster2_proj[:, 0], cluster2_proj[:, 1], s=3, c='#ffff00')
    ax.scatter(targets[:, 0], targets[:, 1], s=12, c='#ff0000')

    fig.savefig(pjoin(mydir, 'fig2_proj.png'), transparent=True, dpi=200)





if __name__ == '__main__':
    os.makedirs(SAVEDIR, exist_ok=True)
    fig2()