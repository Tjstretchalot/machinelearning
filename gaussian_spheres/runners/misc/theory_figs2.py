"""More theory figures
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from os.path import join as pjoin
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import shared.filetools

SAVEDIR = shared.filetools.savepath()

DOTCOL = '#000085'

mpl.rcParams.update({'font.size': 30})

def _plotting_in_higher_dims(dpi=80, **kwargs):
    fig, ax = plt.subplots()

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([-1, -0.5, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0.5, 1])
    ax.set_xticklabels(['-1', '-.5', '.5', '1'])
    ax.set_yticklabels(['-1', '-.5', '.5', '1'])

    ax.text(0.98, 0.01, 'x', color='black')
    ax.text(0.01, 0.98, 'y', color='black')

    if 'fig1' in kwargs and kwargs['fig1']:
        fig.savefig(pjoin(SAVEDIR, 'fig1.png'), dpi=dpi)

    ax.text(0.3, 0.7, '$\\langle -\\frac{1}{2}, \\frac{1}{2} \\rangle$', color=DOTCOL, fontsize=48)

    if 'fig2' in kwargs and kwargs['fig2']:
        fig.savefig(pjoin(SAVEDIR, 'fig2.png'), dpi=dpi)

    ax.scatter([-0.5], [0.5], c=DOTCOL, s=160)

    if 'fig3' in kwargs and kwargs['fig3']:
        fig.savefig(pjoin(SAVEDIR, 'fig3.png'), dpi=dpi)

    plt.close(fig)

def _plotting_in_higher_dims2(dpi=80, **kwargs):
    fig, ax = plt.subplots()

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(['-1', '-.5', '0', '.5', '1'])
    ax.set_yticklabels(['-1', '-.5', '0', '.5', '1'])

    ax.text(0.98, -0.98, 'x', color='black', fontsize=48)
    ax.text(-0.98, 0.98, 'y', color='black', fontsize=48)

    if 'fig4' in kwargs and kwargs['fig4']:
        fig.savefig(pjoin(SAVEDIR, 'fig4.png'), dpi=dpi)

    ax.text(0.3, 0.7, '$\\langle -\\frac{1}{2}, \\frac{1}{2} \\rangle$', color=DOTCOL, fontsize=48)

    if 'fig5' in kwargs and kwargs['fig5']:
        fig.savefig(pjoin(SAVEDIR, 'fig5.png'), dpi=dpi)

    ax.scatter([-0.5], [0.5], c=DOTCOL, s=160)

    if 'fig6' in kwargs and kwargs['fig6']:
        fig.savefig(pjoin(SAVEDIR, 'fig6.png'), dpi=dpi)

def _plotting_in_higher_dims3(dpi=80, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for axis in ('x', 'y', 'z'):
        getattr(ax, f'set_{axis}lim')(-1, 1)
        getattr(ax, f'set_{axis}ticks')([-1, -0.5, 0, 0.5, 1])
        getattr(ax, f'set_{axis}ticklabels')(['-1', '', '0', '', '1'])

    ax.text2D(0.25, -0.09, 'x', transform=ax.transAxes, color='black', fontsize=48)
    ax.text2D(0.83, 0, 'y', transform=ax.transAxes, color='black', fontsize=48)
    ax.text2D(1, 0.5, 'z', transform=ax.transAxes, color='black', fontsize=48)

    if 'fig7' in kwargs and kwargs['fig7']:
        fig.savefig(pjoin(SAVEDIR, 'fig7.png'), dpi=dpi)

    ax.text(-1, 1, 1, '$\\langle -\\frac{1}{2}, \\frac{1}{2}, -1 \\rangle$', color=DOTCOL, fontsize=48)
    #ax.text(-1, 1, 1, '$\\langle -x_1, x_2, x_3 \\rangle$', color=DOTCOL, fontsize=48)

    if 'fig8' in kwargs and kwargs['fig8']:
        fig.savefig(pjoin(SAVEDIR, 'fig8.png'), dpi=dpi)

    ax.scatter([-0.5], [0.5], [-1], c=DOTCOL, s=40)
    #ax.scatter([0.3], [0.5], [0.7], c=DOTCOL, s=160)

    if 'fig9' in kwargs and kwargs['fig9']:
        fig.savefig(pjoin(SAVEDIR, 'fig9.png'), dpi=dpi)

    for i in range(6):
        letter = chr(ord('a') + i)
        z = -1 + 0.2 * (i+1)
        textx = 0 #if i < 3 else 0.1
        texty = 1 - 0.12*i
        ax.text2D(textx, texty, '$\\langle -\\frac{1}{2}, \\frac{1}{2}, ' + f'{z:.1f}' + ' \\rangle$', transform=ax.transAxes, fontsize=24)
        ax.scatter([-0.5], [0.5], [z], c='black', s=40, alpha=0.8)

        if 'fig10' in kwargs and kwargs['fig10']:
            fig.savefig(pjoin(SAVEDIR, f'fig10{letter}.png'), dpi=dpi)


def _all(dpi=80):
    _plotting_in_higher_dims(fig1=True, fig2=True, fig3=True, dpi=dpi)
    _plotting_in_higher_dims2(fig4=True, fig5=True, fig6=True, dpi=dpi)
    _plotting_in_higher_dims3(fig7=True, fig8=True, fig9=True, fig10=True, dpi=dpi)
if __name__ == '__main__':
    mpl.rc('text', usetex=True)
    os.makedirs(SAVEDIR, exist_ok=True)
    #_plotting_in_higher_dims3(fig9=True, dpi=400)
    _all(dpi=400)