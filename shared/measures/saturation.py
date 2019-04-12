"""This module measures the levels of activation that the network is achieving.
One use for this is determining if a sigmoid activation such as tanh is saturated."""

import typing
import numpy as np

import shared.measures.utils as mutils
import matplotlib.pyplot as plt

import os
from shared.filetools import zipdir

class SaturationTrajectory:
    """Contains some meaningful statistics about values of the hidden activations
    through layers.

    Attributes:
        flattened (list[np.ndarray]) a num_layers long list of the flattened and sorted (asc)
            hidden activations
    """

    def __init__(self, flattened: typing.List[np.ndarray]):
        if not isinstance(flattened, (list, tuple)):
            raise ValueError(f'expected flattened is list or tuple, got {flattened}')

        for idx, arr in enumerate(flattened):
            mutils.verify_ndarray(arr, f'flattened[{idx}]', (None,), 'float')

        self.flattened = flattened

BUCKETING_TECHNIQUES = {
    'auto': 'max(Freedman Diaconis Estimator, sturges)',
    'fd': 'Freedman Diaconis Estimator',
    'sturges': 'Sturges',
    'rice': 'Rice',
    'sqrt': 'Sqrt'
}

def measure(hacts: mutils.NetworkHiddenActivations) -> SaturationTrajectory:
    """Measures the saturation trajectory for the given hidden activations

    Args:
        hacts (mutils.NetworkHiddenActivations): the activations from the network

    Returns:
        SaturationTrajectory: the plottable saturation info
    """
    hacts.numpy()

    flattened = []

    for og_hid_acts in hacts.hid_acts:
        hid_acts_flat = np.abs(og_hid_acts.flatten())
        hid_acts_flat.sort()
        flattened.append(hid_acts_flat)

    return SaturationTrajectory(flattened)


def _plot_boxplot(traj: SaturationTrajectory, outfile: str, xlabel: str):
    fig, ax = plt.subplots()
    ax.set_title(f'Absolute Hidden Activations through {xlabel}')
    ax.boxplot(traj.flattened)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Absolute Hidden Activation')
    fig.savefig(outfile)
    plt.close(fig)

def _plot_hist(traj: SaturationTrajectory, outfile: str, xlabel: str, technique: str):
    fig, axs = plt.subplots(nrows=1, ncols=len(traj.flattened), figsize=(len(traj.flattened)*7, 4.8))

    for idx, hid_acts in enumerate(traj.flattened):
        ax = axs[idx]
        ax.hist(hid_acts, bins=technique, density=1)
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Absolute Hidden Activations')
        ax.set_title(f'{xlabel} {idx}')

    fig.savefig(outfile)
    plt.close(fig)

def plot(traj: SaturationTrajectory, outfile: str, exist_ok: bool = False,
         xlabel: str = 'Layers') -> None:
    """Plots saturation information through layers to the given folder

    Args:
        traj (SaturationTrajectory): the trajectory to plot
        outfile (str): the zip file to save plots to
        exist_ok (bool, optional): Defaults to False. True to overwrite, False to error
            if the file already exists
        xlabel (str, optional): Defaults to 'Layers'. The label for the x-axis for plots
            that go through layers
    """

    outfile, outfile_wo_ext = mutils.process_outfile(outfile, exist_ok)

    os.makedirs(outfile_wo_ext)
    _plot_boxplot(traj, os.path.join(outfile_wo_ext, 'boxplot.png'), xlabel)
    for identifier in BUCKETING_TECHNIQUES:
        _plot_hist(traj, os.path.join(outfile_wo_ext, f'hist_{identifier}.png'), xlabel, identifier)

    if exist_ok and os.path.exists(outfile):
        os.remove(outfile)
    zipdir(outfile_wo_ext)

digest = mutils.digest_std(measure, plot) # pylint: disable=invalid-name
during_training = mutils.during_training_std('SATUR', measure, plot, 'shared.measures.saturation') # pylint: disable=invalid-name
