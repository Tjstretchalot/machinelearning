"""This is a measure of how effective an SVM trained with the hidden
activations performs at a task as you go across layers or time.
"""

import torch
import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt

import typing
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations
from shared.pwl import PointWithLabelProducer
from shared.filetools import zipdir
from shared.trainer import GenericTrainingContext
import shared.npmp as npmp
import shared.measures.utils as mutils

import os

class SVMTrajectory(typing.NamedTuple):
    """Describes a trajectory of svm performance over time.

    Attributes:
        overall (torch.tensor [num_layers]): a float-like tensor of the SVM
            accuracy through layers for correctly identifying the class

        by_label_vs_all (torch.tensor[num_layers, output_dim], optional):
            a float-like tensor of the SVM accuracy through layers. for each output label an svm
            is trained to classify "is that label" vs "is not that label". Should be None if there
            are only two input labels
    """
    overall: torch.tensor
    by_label_vs_all: typing.Optional[torch.tensor]

def train_svms_with(sample_points: torch.tensor, sample_labels: torch.tensor,
                    *hidden_states: typing.Tuple[torch.tensor]) -> SVMTrajectory:
    """Produces as svm trajectory from the given sample points, labels, and
    hidden activations

    Args:
        sample_points (torch.tensor): The points that were sent through the network
        sample_labels (torch.tensor): The labels that were sent through the network
        hid_acts (tuple[torch.tensor]): the hidden activations that arose out of the points
    """

    num_points = sample_points.shape[0]
    train_points = int(num_points * (13.0/15.0))
    output_dim = hidden_states[-1].shape[1]

    overall = []

    if output_dim > 2:
        masks = [sample_labels == lbl for lbl in range(output_dim)]
        by_label_vs_all = []

    for state in hidden_states(sample_points):
        if not torch.is_tensor(state):
            raise ValueError(f'expected hidden state is tensor, got {state} (type={type(state)})')
        if len(state.shape) != 2:
            raise ValueError(f'expected state has shape [batch_size, layer_size] but has shape {state.shape}')
        if state.shape[0] != num_points:
            raise ValueError(f'expected state has shape [batch_size={num_points}, layer_size] but has shape {state.shape}')
        if state.dtype == torch.float:
            state = state.double()
        elif state.dtype != torch.double:
            raise ValueError(f'expected state has dtype float-like, but has {state.dtype}')

        classifier = svm.LinearSVC(max_iter=5000)
        classifier.fit(state[:train_points], sample_labels[:train_points])
        mean_acc = classifier.score(state[train_points:], sample_labels[train_points:])

        if not isinstance(mean_acc, float):
            raise ValueError(f'expected mean_acc is float, got {mean_acc} (type={type(mean_acc)})')

        overall.append(mean_acc)

        if output_dim > 2:
            layer_accs = []
            for lbl in range(output_dim):
                classifier = svm.LinearSVC(5000)
                classifier.fit(state[:train_points], masks[lbl][:train_points])
                mean_acc = classifier.score(state[train_points:], masks[lbl][train_points:])
                layer_accs.append(mean_acc)
            by_label_vs_all.append(layer_accs)

    if output_dim <= 2:
        return SVMTrajectory(overall=torch.tensor(overall, dtype=torch.double), by_label_vs_all=None)
    else:
        return SVMTrajectory(overall=torch.tensor(overall, dtype=torch.double),
                             by_label_vs_all=torch.tensor(by_label_vs_all, dtype=torch.double))


def train_svms(pwl_prod: PointWithLabelProducer,
               hidden_states: typing.Callable) -> SVMTrajectory:
    """Trains svms on each layer. This is invariant to feed-forward or
    recurrent layers; hidden_states should be a function which accepts
    a tensor [batch_size, input_dim] and returns an iterable of tensors
    [batch_size, layer_size]. Each layer size may be different, however
    each call to hidden_states must result in exactly the same number of
    layers in the same order.

    Args:
        pwl_prod (PointWithLabelProducer): the problem to show to the network
        hidden_states (typing.Callable): a function which gives an iterable of the
            hidden states when given a points tensor

    Returns:
        traj (SVMTrajectory): the svm accuracy through time/layers
    """

    num_points = min(pwl_prod.output_dim * 150, pwl_prod.epoch_size)

    sample_points = torch.zeros((num_points, pwl_prod.input_dim), dtype=torch.double)
    sample_labels = torch.zeros(num_points, dtype=torch.long)

    pwl_prod.fill(sample_points, sample_labels)

    hid_acts = tuple(state for state in hidden_states(sample_points))
    return train_svms_with(sample_points, sample_labels, *hid_acts)

def train_svms_ff(network: FeedforwardNetwork,
                  pwl_prod: PointWithLabelProducer) -> SVMTrajectory:
    """Produces the svm trajectory of the given feed forward network

    Args:
        network (FeedforwardNetwork): the network to evaluate
        pwl_prod (PointWithLabelProducer): the points producer

    Returns:
        traj (SVMTrajectory): the trajectory found
    """

    def hidden_states(points):
        result = []

        def on_hidacts(acts_info: FFHiddenActivations):
            result.append(acts_info.hidden_acts.detach())

        network(points, on_hidacts)

        return result

    return train_svms(pwl_prod, hidden_states)

def plot_traj_ff(traj: SVMTrajectory, outfile: str, exist_ok: bool = False):
    """Plots the given trajectory to the given file. The file should have no extension or
    have the .zip extension.

    Args:
        traj (SVMTrajectory): the trajectory to plot
        outfile (str): where to save the plot (will be zipped)
        exist_ok (bool, optional): Defaults to False. if existing files should be overwritten
    """
    if not isinstance(traj, SVMTrajectory):
        raise ValueError(f'expected traj is SVMTrajectory, got {traj}')
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile}')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok}')

    outfile_wo_ext = os.path.splitext(outfile)[0]
    if outfile == outfile_wo_ext:
        outfile += '.zip'

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'need {outfile_wo_ext} as working space to create {outfile}')

    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'{outfile} already exists (use exist_ok=True to overwrite)')

    os.makedirs(outfile_wo_ext)

    xlabel = 'Layers'
    ylabel = 'SVM Accuracy (%)'

    layers = np.arange(traj.overall.shape[0])
    num_labels = int(traj.by_label_vs_all.shape[1]) if traj.by_label_vs_all is not None else 2
    chance_perc = 1.0 / num_labels

    fig, ax = plt.subplots()

    ax.set_title(f'{ylabel} Through {xlabel} (Overall)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if traj.by_label_vs_all is not None:
        for lbl in range(traj.by_label_vs_all.shape[1]):
            ax.plot(layers, traj.by_label_vs_all[:, lbl].numpy(), linestyle='dashed', label=f'{lbl} vs all', alpha=0.6)
    ax.plot(layers, traj.overall.numpy(), label='Overall')
    ax.set_xticks(layers)
    ax.legend(loc=1)

    fig.savefig(os.path.join(outfile_wo_ext, 'overall.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_title(f'{ylabel} Through {xlabel} (All Only)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(chance_perc, layers.min().item(), layers.max().item(), linestyle='dashed', color='k', label='Chance Acc.', alpha=0.6)
    ax.plot(layers, traj.overall.numpy(), label='Overall')
    ax.set_xticks(layers)
    ax.legend(loc=1)

    fig.savefig(os.path.join(outfile_wo_ext, 'allonly.png'))
    plt.close(fig)

    fig, ax = plt.subplots() # previous plot with consistent scale
    ax.set_title(f'{xlabel} Through {ylabel} (All Only)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(chance_perc, layers.min(), layers.max(), linestyle='dashed', color='k', label='Chance Acc.', alpha=0.6)
    ax.plot(layers, traj.overall.numpy(), label='Overall')
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    ax.legend(loc=1)

    fig.savefig(os.path.join(outfile_wo_ext, 'allonly_0_1_scale.png'))
    plt.close(fig)

    if traj.by_label_vs_all is not None:
        best_square = int(np.ceil(np.sqrt(num_labels)))
        num_cols = best_square
        num_rows = int(np.ceil(num_labels / best_square))
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, sharey='all', sharex='all')
        chance_perc = (num_labels - 1) / num_labels

        fig.suptitle(f'{xlabel} through {ylabel} (By Label 1 vs All)')
        lbl = 0
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axes[row][col]
                if lbl >= num_labels:
                    ax.remove()
                    continue

                yvals = traj.by_label_vs_all[:, lbl].numpy()
                ax.set_title(str(lbl))
                ax.plot(layers, yvals, label=str(lbl))
                ax.axhline(chance_perc, layers.min(), layers.max(), linestyle='dashed', color='k', label='Chance Acc.', alpha=0.6)
                lbl += 1

        axes[0][0].set_xticks(layers)
        fig.savefig(os.path.join(outfile_wo_ext, 'by_label.png'))
        plt.close(fig)

    if exist_ok and os.path.exists(outfile):
        os.remove(outfile)
    zipdir(outfile_wo_ext)

def digest_train_and_plot_ff(sample_points: torch.tensor, sample_labels: torch.tensor,
                              *all_hid_acts: typing.Tuple[torch.tensor],
                              savepath: str = None):
    """Digestor friendly way to find the svmtrajectory and then plot it at the given
    savepath for the given sample points, labels, and hidden activations"""
    if savepath is None:
        raise ValueError(f'expected savepath is str, got {savepath} (type={type(savepath)})')

    traj = train_svms_with(sample_points, sample_labels, *all_hid_acts)
    plot_traj_ff(traj, savepath)

def during_training_ff(savepath: str, train: bool,
                       digestor: typing.Optional[npmp.NPDigestor] = None):
    """Returns a callable that is good for OnEpochCaller or similar intermediaries
    that saves the svm information to the given directory. The filenames will be
    svm_{hint}.zip. This expects that savepath will not exist.

    Args:
        savepath (str): the folder to save to
        train (bool): true for training data, false for test data
        digestor (NPDigestor, optional): if specified, used for parallelization

    Returns:
        a callable that accepts a GenericTrainingContext
    """

    if os.path.exists(savepath):
        raise ValueError(f'{savepath} already exists')
    if digestor is not None and not isinstance(digestor, npmp.NPDigestor):
        raise ValueError(f'expected digestor is NPDigestor, got {digestor} (type={type(digestor)})')

    def on_step(context: GenericTrainingContext, fname_hint: str):
        context.logger.info('[SVM] Measuring SVM Through Layers (hint: %s)', fname_hint)
        pwl = context.train_pwl if train else context.test_pwl

        if digestor is not None:
            num_points = min(150*pwl.output_dim, pwl.epoch_size)
            hacts = mutils.get_hidacts_ff(context.model, pwl, num_points)
            digestor(hacts.sample_points, hacts.sample_labels, *hacts.hid_acts,
                     target_module='shared.measures.svm',
                     target_name='digest_train_and_plot_ff',
                     savepath=savepath)
            return
        traj = train_svms_ff(context.model, pwl)
        plot_traj_ff(traj, os.path.join(savepath, f'svm_{fname_hint}'), False)

    return on_step

