"""Measures distances of points from different labels across time. This is
an rnn-specific measure of dimensionality.

One particularly interesting situation would be if the RNN started with
the labels and within-labels separated, spread them out for sorting, then
they merged back down such that points with the same label looked similar.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import typing
import logging

from shared.models.rnn import NaturalRNN, RNNHiddenActivations
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations
from shared.pwl import PointWithLabelProducer
from shared.filetools import zipdir, unzip
from shared.trainer import GenericTrainingContext

import scipy.spatial.distance as sdist

import os
import shutil

def measure_instant(hid_acts: torch.tensor,
                    labels: torch.tensor,
                    num_labels: int) -> typing.Tuple[float, float]:
    """Returns the mean distance of points with the same label from each other
    as the first result, and the mean distance of points with different labels
    from each other as the second result.

    Args:
        hid_acts (torch.tensor): a (number_samples, hidden_dim) tensor of the activations
            of the network at a particular point in time
        labels (torch.tensor): a (number_samples) tensor of the labels for the points
            in hid_acts
        num_labels (int): the number of labels
    Returns:
        within_dists (torch.tensor): a 1d array of the within-point distances
        across_dists (torch.tensor): a 1d array of the across-point distances
    """
    if not torch.is_tensor(hid_acts):
        raise ValueError(f'expected hid_acts to be tensor, got {hid_acts} (type={type(hid_acts)})')
    if len(hid_acts.shape) != 2:
        raise ValueError(f'expected hid_acts to have shape (num_samples, hidden_dim), but has shape {hid_acts.shape}')
    if hid_acts.dtype not in (torch.double, torch.float):
        raise ValueError(f'expected hid_acts to have dtype double, has dtype {hid_acts.dtype}')
    if not torch.is_tensor(labels):
        raise ValueError(f'expected labels to be tensor, got {labels} (type={type(labels)})')
    if len(labels.shape) != 1:
        raise ValueError(f'expected labels to have shape (num_samples) but has shape {labels.shape}')
    if labels.dtype not in (torch.uint8, torch.int, torch.long):
        raise ValueError(f'expected labels to have dtype long, has dtype {labels.dtype}')
    if labels.shape[0] != hid_acts.shape[0]:
        raise ValueError(f'expected hid_acts to have shape (number_samples, hidden_dim) and labels to have shape (number_samples), but hid_acts.shape={hid_acts.shape} and labels.shape={labels.shape} (dont match on dim 0)')

    for lbl in range(num_labels):
        if (labels == lbl).sum() <= 5:
            raise ValueError(f'not enough points with label {lbl} (got {(labels == lbl).sum()})')

    within_dists = None
    across_dists = None
    for lbl in range(num_labels):
        pairwise_dists = sdist.pdist(hid_acts[labels == lbl].numpy())
        within_dists = np.concatenate([within_dists, pairwise_dists]) if within_dists is not None else pairwise_dists
        for lbl2 in range(lbl + 1, num_labels):
            pairwise_dists = sdist.cdist(hid_acts[labels == lbl].numpy(), hid_acts[labels == lbl2].numpy()).flatten()
            across_dists = np.concatenate([across_dists, pairwise_dists]) if across_dists is not None else pairwise_dists

    return torch.from_numpy(within_dists), torch.from_numpy(across_dists)


def _dbg(verbose: bool, logger: typing.Optional[logging.Logger], msg: str):
    if not verbose:
        return
    if logger:
        logger.debug(msg)
    else:
        print(msg)

def measure_dtt(model: NaturalRNN, pwl_prod: PointWithLabelProducer,
                duration: int, outfile: str, exist_ok: bool = False,
                logger: logging.Logger = None, verbose: bool = False) -> None:
    """Measures the distance of points in hidden activation space from points which
    are sampled from different labels. For example, points from the same label might
    be close in hidden activation space even when they are far in input space.

    Args:
        model (NaturalRNN): the model
        pwl_prod (PointWithLabelProducer): the points to test
        duration (int): the number of recurrent times
        outfile (str): where to save the result. Should have extension '.zip' or
                       no extension at all
        exist_ok (bool, default false): true if we should overwrite the outfile if it already
            exists, false to check if it exists and error if it does
    """

    if not isinstance(model, NaturalRNN):
        raise ValueError(f'expected model is NaturalRNN, got {model} (type={type(model)})')
    if not isinstance(pwl_prod, PointWithLabelProducer):
        raise ValueError(f'expected pwl is PointWithLabelProducer, got {pwl_prod} (type=({type(pwl_prod)})')
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')

    outfile_wo_ext = os.path.splitext(outfile)[0]
    if outfile_wo_ext == outfile:
        outfile = outfile_wo_ext + '.zip'

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'for outfile={outfile}, need {outfile_wo_ext} as working space')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'outfile {outfile} already exists (use exist_ok=True) to overwrite')


    num_samples = min(pwl_prod.epoch_size, 50 * pwl_prod.output_dim)
    sample_points = torch.zeros((num_samples, model.input_dim), dtype=torch.double)
    sample_labels = torch.zeros((num_samples,), dtype=torch.long)
    hid_acts = torch.zeros((duration+1, num_samples, model.hidden_dim), dtype=torch.double)
    within_dists = [] # each value corresponds to a torch tensor of within dists
    within_means = torch.zeros(duration+1, dtype=torch.double)
    within_stds = torch.zeros(duration+1, dtype=torch.double)
    within_sems = torch.zeros(duration+1, dtype=torch.double)
    across_dists = [] # each value corresponds to a torch tensor of across dists
    across_means = torch.zeros(duration+1, dtype=torch.double)
    across_stds = torch.zeros(duration+1, dtype=torch.double)
    across_sems = torch.zeros(duration+1, dtype=torch.double)

    pwl_prod.fill(sample_points, sample_labels)

    def on_hidacts(acts_info: RNNHiddenActivations):
        hidden_acts = acts_info.hidden_acts
        recur_step = acts_info.recur_step

        hid_acts[recur_step, :, :] = hidden_acts.detach()

        within, across = measure_instant(hid_acts[recur_step], sample_labels, pwl_prod.output_dim)
        within_dists.append(within)
        across_dists.append(across)

        within_means[recur_step] = within.mean()
        within_stds[recur_step] = within.std()
        within_sems[recur_step] = within_stds[recur_step] / np.sqrt(num_samples)

        across_means[recur_step] = across.mean()
        across_stds[recur_step] = across.std()
        across_sems[recur_step] = across_stds[recur_step] / np.sqrt(num_samples)

    _dbg(verbose, logger, 'measure_dtt getting raw data')
    model(sample_points, duration, on_hidacts, 1)


    within_col, across_col = 'tab:cyan', 'r'

    fig_mean_with_stddev, ax_mean_with_stddev = plt.subplots()
    fig_mean_with_sem, ax_mean_with_sem = plt.subplots()
    fig_mean_with_scatter, ax_mean_with_scatter = plt.subplots()

    ax_mean_with_stddev.set_title('Distances Through Time (error: 1.96 std dev)')
    ax_mean_with_sem.set_title('Distances Through Time (error: 1.96 sem)')
    ax_mean_with_scatter.set_title('Distances Through Time')

    for ax in (ax_mean_with_stddev, ax_mean_with_sem, ax_mean_with_scatter):
        ax.set_xlabel('Time (recurrent steps occurred)')
        ax.set_ylabel('Distance (euclidean)')

    recur_steps = np.arange(duration+1)
    _dbg(verbose, logger, 'measure_dtt plotting mean_with_stddev')
    ax_mean_with_stddev.errorbar(recur_steps, within_means.numpy(), within_stds.numpy() * 1.96, color=within_col, label='Within')
    ax_mean_with_stddev.errorbar(recur_steps, across_means.numpy(), across_stds.numpy() * 1.96, color=across_col, label='Across')
    _dbg(verbose, logger, 'measure_dtt plotting mean_with_sem')
    ax_mean_with_sem.errorbar(recur_steps, within_means.numpy(), within_sems.numpy() * 1.96, color=within_col, label='Within')
    ax_mean_with_sem.errorbar(recur_steps, across_means.numpy(), across_sems.numpy() * 1.96, color=across_col, label='Across')
    _dbg(verbose, logger, 'measure_dtt plotting mean_with_scatter')
    ax_mean_with_scatter.plot(recur_steps, within_means.numpy(), color=within_col, label='Within')
    ax_mean_with_scatter.plot(recur_steps, across_means.numpy(), color=across_col, label='Across')

    for recur_step in range(duration+1):
        xvals = np.zeros(within_dists[recur_step].shape, dtype='uint8') + recur_step
        ax_mean_with_scatter.scatter(xvals, within_dists[recur_step], 1, within_col, alpha=0.3)
        xvals = np.zeros(across_dists[recur_step].shape, dtype='uint8') + recur_step
        ax_mean_with_scatter.scatter(xvals, across_dists[recur_step], 1, across_col, alpha=0.3)

    _dbg(verbose, logger, 'measure_dtt saving and cleaning up')
    for ax in (ax_mean_with_stddev, ax_mean_with_sem, ax_mean_with_scatter):
        ax.legend()
        ax.set_xticks(recur_steps)

    for fig in (fig_mean_with_stddev, fig_mean_with_sem, fig_mean_with_scatter):
        fig.tight_layout()

    os.makedirs(outfile_wo_ext)

    fig_mean_with_stddev.savefig(os.path.join(outfile_wo_ext, 'mean_with_stddev.png'), transparent=True)
    fig_mean_with_sem.savefig(os.path.join(outfile_wo_ext, 'mean_with_sem.png'), transparent=True)
    fig_mean_with_scatter.savefig(os.path.join(outfile_wo_ext, 'mean_with_scatter.png'), transparent=True)

    plt.close(fig_mean_with_stddev)
    plt.close(fig_mean_with_sem)
    plt.close(fig_mean_with_scatter)

    np.savez(os.path.join(outfile_wo_ext, 'data.npz'),
        sample_points=sample_points.numpy(),
        sample_labels=sample_labels.numpy(),
        hid_acts=hid_acts.numpy()
        )
    np.savez(os.path.join(outfile_wo_ext, 'within.npz'), *within_dists)
    np.savez(os.path.join(outfile_wo_ext, 'across.npz'), *across_dists)

    if os.path.exists(outfile):
        os.remove(outfile)

    cwd = os.getcwd()
    shutil.make_archive(outfile_wo_ext, 'zip', outfile_wo_ext)
    os.chdir(cwd)
    shutil.rmtree(outfile_wo_ext)
    os.chdir(cwd)

def measure_dtt_ff(model: FeedforwardNetwork, pwl_prod: PointWithLabelProducer,
                   outfile: str, exist_ok: bool = False,
                   logger: typing.Optional[logging.Logger] = None,
                   verbose: bool = False) -> None:
    """Analogue to measure_dtt for feed-forward networks"""
    if not isinstance(model, FeedforwardNetwork):
        raise ValueError(f'expected model is FeedforwardNetwork, got {model} (type={type(model)})')
    if not isinstance(pwl_prod, PointWithLabelProducer):
        raise ValueError(f'expected pwl_prod is PointWithLabelProducer, got {pwl_prod} (type={type(pwl_prod)})')
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile}')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok}')
    if logger is not None and not isinstance(logger, logging.Logger):
        raise ValueError(f'expected logger is optional[logging.Logger], got {logger} (type={type(logger)})')
    if not isinstance(verbose, bool):
        raise ValueError(f'expected verbose is bool, got {verbose} (type={type(verbose)})')

    outfile_wo_ext = os.path.splitext(outfile)[0]
    if outfile_wo_ext == outfile:
        outfile = outfile_wo_ext + '.zip'

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'for outfile={outfile}, need {outfile_wo_ext} as working space')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'outfile {outfile} already exists (use exist_ok=True) to overwrite')

    num_samples = min(pwl_prod.epoch_size, 50 * pwl_prod.output_dim)

    sample_points = torch.zeros((num_samples, model.input_dim), dtype=torch.double)
    sample_labels = torch.zeros((num_samples,), dtype=torch.long)
    hid_acts = [] # each will be 2d tensor
    within_dists = [] # each value corresponds to a torch tensor of within dists
    within_means = torch.zeros(model.num_layers+1, dtype=torch.double)
    within_stds = torch.zeros(model.num_layers+1, dtype=torch.double)
    within_sems = torch.zeros(model.num_layers+1, dtype=torch.double)
    across_dists = [] # each value corresponds to a torch tensor of across dists
    across_means = torch.zeros(model.num_layers+1, dtype=torch.double)
    across_stds = torch.zeros(model.num_layers+1, dtype=torch.double)
    across_sems = torch.zeros(model.num_layers+1, dtype=torch.double)

    pwl_prod.fill(sample_points, sample_labels)

    def on_hidacts(acts_info: FFHiddenActivations):
        hidden_acts = acts_info.hidden_acts
        layer = acts_info.layer

        hid_acts.append(hidden_acts.detach())

        within, across = measure_instant(hid_acts[layer], sample_labels, pwl_prod.output_dim)
        within_dists.append(within)
        across_dists.append(across)

        within_means[layer] = within.mean()
        within_stds[layer] = within.std()
        within_sems[layer] = within_stds[layer] / np.sqrt(num_samples)

        across_means[layer] = across.mean()
        across_stds[layer] = across.std()
        across_sems[layer] = across_stds[layer] / np.sqrt(num_samples)

    _dbg(verbose, logger, 'measure_dtt_ff getting raw data')
    model(sample_points, on_hidacts)

    layers = np.arange(model.num_layers+1)

    _plot_dtt_ff(layers, within_means, within_stds, within_sems,
                 across_means, across_stds, across_sems,
                 within_dists, across_dists, outfile_wo_ext,
                 verbose, logger)

    np.savez(os.path.join(outfile_wo_ext, 'sample.npz'),
        sample_points=sample_points.numpy(),
        sample_labels=sample_labels.numpy()
        )
    np.savez(os.path.join(outfile_wo_ext, 'hid_acts.npz'), *hid_acts)
    np.savez(os.path.join(outfile_wo_ext, 'within.npz'), *within_dists)
    np.savez(os.path.join(outfile_wo_ext, 'across.npz'), *across_dists)

    if os.path.exists(outfile):
        os.remove(outfile)

    zipdir(outfile_wo_ext)

def replot_dtt_ff(infile: str, verbose: bool = True, logger: logging.Logger = None):
    """Recreates the dtt_ff plots for the given zip, replacing them inside the zip.

    Args:
        infile (str): the outfile that you used when measuring
        verbose (bool): if this should print progress information
        logger (Logger): the logger to use, None for print
    """
    if not isinstance(infile, str):
        raise ValueError(f'expected infile is str, got {infile} (type={type(infile)})')
    if not isinstance(verbose, bool):
        raise ValueError(f'expected verbose is bool, got {verbose} (type={type(verbose)})')
    if logger is not None and not isinstance(logger, logging.Logger):
        raise ValueError(f'expected logger is optional[logging.Logger], got {logger} (type={type(logger)})')

    _dbg(verbose, logger, f'unpacking {infile}')
    unzip(infile)

    infile_wo_ext = os.path.splitext(infile)[0]
    _dbg(verbose, logger, f'fetching data')
    try:
        within_dists, across_dists = [], []
        num_samples: int

        with np.load(os.path.join(infile_wo_ext, 'within.npz')) as within_dict:
            i = 0
            while f'arr_{i}' in within_dict:
                within_dists.append(within_dict[f'arr_{i}'])
                i += 1

        with np.load(os.path.join(infile_wo_ext, 'across.npz')) as across_dict:
            i = 0
            while f'arr_{i}' in across_dict:
                across_dists.append(across_dict[f'arr_{i}'])
                i += 1

        with np.load(os.path.join(infile_wo_ext, 'sample.npz')) as sample_dict:
            num_samples = sample_dict['sample_labels'].shape[0]

        num_layers = len(within_dists) - 1
        if len(across_dists) != num_layers + 1:
            raise ValueError(f'expected within_dists has same len as across_dists, but len(within_dists)={len(within_dists)}, len(across_dists)={len(across_dists)}')


        within_means = torch.zeros(num_layers+1, dtype=torch.double)
        within_stds = torch.zeros(num_layers+1, dtype=torch.double)
        within_sems = torch.zeros(num_layers+1, dtype=torch.double)
        across_means = torch.zeros(num_layers+1, dtype=torch.double)
        across_stds = torch.zeros(num_layers+1, dtype=torch.double)
        across_sems = torch.zeros(num_layers+1, dtype=torch.double)

        for i in range(num_layers+1):
            within_means[i] = within_dists[i].mean()
            within_stds[i] = within_dists[i].std()
            within_sems[i] = within_stds[i] / np.sqrt(num_samples)
            across_means[i] = across_dists[i].mean()
            across_stds[i] = across_dists[i].std()
            across_sems[i] = across_stds[i] / np.sqrt(num_samples)

        layers = np.arange(num_layers+1)

        _plot_dtt_ff(layers, within_means, within_stds, within_sems,
                    across_means, across_stds, across_sems,
                    within_dists, across_dists, infile_wo_ext,
                    verbose, logger)
    finally:
        _dbg(verbose, logger, f'repacking {infile}')
        zipdir(infile_wo_ext)

def _plot_dtt_ff(layers, within_means, within_stds, within_sems,
                 across_means, across_stds, across_sems,
                 within_dists, across_dists, outfile_wo_ext,
                 verbose, logger):
    within_col, across_col, ratio_col = 'tab:cyan', 'r', 'k'

    fig_mean_with_stddev, ax_mean_with_stddev = plt.subplots()
    fig_mean_with_sem, ax_mean_with_sem = plt.subplots()
    fig_mean_with_scatter, ax_mean_with_scatter = plt.subplots()

    ax_mean_with_stddev.set_title('Distances Through Layers (error: 1.96 std dev)')
    ax_mean_with_sem.set_title('Distances Through Layers (error: 1.96 sem)')
    ax_mean_with_scatter.set_title('Distances Through Layers')

    _dbg(verbose, logger, 'plotting mean_with_stddev')
    ax_mean_with_stddev.errorbar(layers, within_means.numpy(), within_stds.numpy() * 1.96, color=within_col, label='Within')
    ax_mean_with_stddev.errorbar(layers, across_means.numpy(), across_stds.numpy() * 1.96, color=across_col, label='Across')
    _dbg(verbose, logger, 'plotting mean_with_sem')
    ax_mean_with_sem.errorbar(layers, within_means.numpy(), within_sems.numpy() * 1.96, color=within_col, label='Within')
    ax_mean_with_sem.errorbar(layers, across_means.numpy(), across_sems.numpy() * 1.96, color=across_col, label='Across')
    _dbg(verbose, logger, 'plotting mean_with_scatter')
    ax_mean_with_scatter.plot(layers, within_means.numpy(), color=within_col, label='Within')
    ax_mean_with_scatter.plot(layers, across_means.numpy(), color=across_col, label='Across')

    for lay in layers:
        xvals = np.zeros(within_dists[lay].shape, dtype='uint8') + lay
        ax_mean_with_scatter.scatter(xvals, within_dists[lay], 1, within_col, alpha=0.3)
        xvals = np.zeros(across_dists[lay].shape, dtype='uint8') + lay
        ax_mean_with_scatter.scatter(xvals, across_dists[lay], 1, across_col, alpha=0.3)

    _dbg(verbose, logger, 'plotting ratios')
    ratios = within_means.clone() / across_means
    for ax in (ax_mean_with_stddev, ax_mean_with_sem, ax_mean_with_scatter):
        twinned = ax.twinx()
        twinned.plot(layers, ratios.numpy(), linestyle='dashed', color=ratio_col, label='Within/Across', alpha=0.8)
        twinned.legend(loc=2)
        twinned.set_ylabel('Within / Across (ratio)')
        ax.set_ylabel('Distances (Euclidean)')
        ax.set_xlabel('Layer')
        ax.legend(loc=1)
        ax.set_xticks(layers)

    _dbg(verbose, logger, 'saving and cleaning up')
    for fig in (fig_mean_with_stddev, fig_mean_with_sem, fig_mean_with_scatter):
        fig.tight_layout()

    os.makedirs(outfile_wo_ext, exist_ok=True)

    fig_mean_with_stddev.savefig(os.path.join(outfile_wo_ext, 'mean_with_stddev.png'), transparent=True)
    fig_mean_with_sem.savefig(os.path.join(outfile_wo_ext, 'mean_with_sem.png'), transparent=True)
    fig_mean_with_scatter.savefig(os.path.join(outfile_wo_ext, 'mean_with_scatter.png'), transparent=True)

    plt.close(fig_mean_with_stddev)
    plt.close(fig_mean_with_sem)
    plt.close(fig_mean_with_scatter)

def during_training_ff(savepath: str, train: bool, **kwargs):
    """Fetches the on_step/on_epoch for things like OnEpochsCaller
    that saves into the given directory.

    Args:
        savepath (str): where to save
        train (bool): true to use training data, false to use validation data
        kwargs (dict): passed to measure_dtt_ff
    """
    if not isinstance(savepath, str):
        raise ValueError(f'expected savepath is str, got {savepath} (type={type(savepath)})')
    if not isinstance(train, bool):
        raise ValueError(f'expected train is bool, got {train} (type={type(train)})')

    if os.path.exists(savepath):
        raise ValueError(f'{savepath} already exists')

    def on_step(context: GenericTrainingContext, fname_hint: str):
        context.logger.info('[DTT] Measuring DTT Through Layers (hint: %s)', fname_hint)
        pwl = context.train_pwl if train else context.test_pwl
        measure_dtt_ff(context.model, pwl, os.path.join(savepath, f'dtt_{fname_hint}', **kwargs))

    return on_step