"""Produces plots of participation ratios, either through time or through
layers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import typing
import os
import time
import sys
import json
import shared.measures.pca as pca
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations
from shared.pwl import PointWithLabelProducer
from shared.filetools import zipdir, unzip
from shared.trainer import GenericTrainingContext
import shared.npmp as npmp
import shared.measures.utils as mutils
import shared.myqueue as myq

import uuid

def measure_pr(hidden_acts: torch.tensor) -> float:
    """Measures the participation ratio of the specified hidden acts, plotting
    it as necessary and saving it to the zip file in the given directory.

    Args:
        hidden_acts (torch.tensor): a [batch_size x layer_size] tensor of activations,
            where the first index tells us which point and the second index tells us
            which neuron. float-like
    """
    if not torch.is_tensor(hidden_acts):
        raise ValueError(f'expected hidden_acts is tensor, got {hidden_acts} (type={type(hidden_acts)})')
    if len(hidden_acts.shape) != 2:
        raise ValueError(f'expected hidden_acts.shape is (batch_size, layer_size), got {hidden_acts.shape}')
    if hidden_acts.dtype not in (torch.float, torch.double):
        raise ValueError(f'expected hidden_acts is float-like, but dtype is {hidden_acts.dtype}')

    eigs = pca.get_hidden_pcs(hidden_acts, None, False)
    result = torch.pow(torch.sum(eigs), 2) / torch.sum(torch.pow(eigs, 2))
    result = result.item()

    if not isinstance(result, float):
        raise ValueError(f'expected participation_ratio is float, got {result} (type={type(result)})')
    if result < 1 - 1e-8:
        raise ValueError(f'PR should always be >= 1 (got {result})')
    if result > eigs.shape[0] + 1e-8:
        raise ValueError(f'PR should always be <= M where M = eigs.shape[0] = {eigs.shape[0]} (got {result})')

    return result

def measure_pr_np(hidden_acts: np.ndarray, iden: typing.Any, outqueue: typing.Any):
    """Measure pr but set up to send the output through a zeromq queue

    Args:
        hidden_acts (np.ndarray): the hidden activations you want the participation ratio for
        iden (any): sent as the first value in the tuple pushed to the outqueue
        outqueue (any): deserialized with myqueue
    """

    if not isinstance(hidden_acts, np.ndarray):
        raise ValueError(f'expected hidden_acts is numpy array, got {hidden_acts} (type={type(hidden_acts)})')

    sys.stdout.flush()
    result = measure_pr(torch.from_numpy(hidden_acts))
    outqueue = myq.ZeroMQQueue.deser(outqueue)
    outqueue.put((iden, result))
    outqueue.close()

class PRTrajectory:
    """Describes the trajectory of participation ratio through time or layers

    Attributes:
        overall (torch.tensor[number layers / recurrent times + 1])
        by_label (list[torch.tensor[number layers / recurrent_times + 1]])
        layers (bool): true if this is a through-layers plot, false if through-time
    """
    overall: torch.tensor
    by_label: typing.List[torch.tensor]
    layers: bool

    def __init__(self, overall, by_label, layers):
        self.overall = overall
        self.by_label = by_label
        self.layers = layers

    def save(self, outfile: str, exist_ok=False):
        """Saves this trajaectory to the given file

        Args:
            outfile (str): the filename to save to; should be a zip file
            exist_ok (bool): True to overwrite outfile if it exists, False not to
        """
        _, folder = mutils.process_outfile(outfile, exist_ok=exist_ok)
        os.makedirs(folder, exist_ok=True)

        meta_dict = {'layers': self.layers}
        with open(os.path.join(folder, 'meta.json'), 'w') as metaout:
            json.dump(meta_dict, metaout)
        torch.save(self.overall, os.path.join(folder, 'overall.pt'))
        if self.by_label is not None:
            torch.save(self.by_label, os.path.join(folder, 'by_label.pt'))
        zipdir(folder)

    @classmethod
    def load(cls, infile: str):
        """Loads the PR trajectory saved to the given filepath

        Arguments:
            infile (str): the filename to load from; should be a zip file
        """
        filename, folder = mutils.process_outfile(infile, exist_ok=True)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        unzip(filename)

        with open(os.path.join(folder, 'meta.json'), 'r') as meta_in:
            meta_dict = json.load(meta_in)
        overall = torch.load(os.path.join(folder, 'overall.pt'))
        by_label = None
        if os.path.exists(os.path.join(folder, 'by_label.pt')):
            by_label = torch.load(os.path.join(folder, 'by_label.pt'))
        zipdir(folder)
        return cls(overall=overall, layers=meta_dict['layers'], by_label=by_label)

class AveragedPRTrajectory:
    """Describes a pr trajectory that is comprised of multiple sub-trajectories that
    come from different realizations of noise or weight initialization

    Attributes:
        trajectories (list[PRTrajectory]): the actual trajectories in this trajectory
        overall (torch.tensor[number of layers]): the averaged tensor overall
        overall_std (torch.tensor[number of layers]): the std error at each layer
        overall_sem (torch.tensor[number of layers]): the std error of mean at each layer

        layers (bool): if this is an average of layer-trajectories
    """

    def __init__(self, trajectories: typing.List[PRTrajectory]):
        if not isinstance(trajectories, (list, tuple)):
            raise ValueError(f'expected trajectories is list[PRTrajectory], got {trajectories} (type={type(trajectories)})')
        if not trajectories:
            raise ValueError(f'need at least one trajectory to average over')
        self.layers = trajectories[0].layers
        for i, val in enumerate(trajectories):
            if not isinstance(val, PRTrajectory):
                raise ValueError(f'expected trajectories[{i}] is PRTrajectory, got {val} (type={type(val)})')
            if val.layers != self.layers:
                raise ValueError(f'got trajectories[0].layers = {self.layers}, trajectories[{i}].layers = {val.layers}')
            if val.overall.shape != trajectories[0].overall.shape:
                raise ValueError(f'got trajectories[0].overall.shape = {trajectories[0].overall.shape}, trajectories[{i}].overall.shape = {val.overall.shape}')

        self.trajectories = trajectories

        self.layers = self.trajectories[0].layers

        overalls = tuple(traj.overall.reshape(-1, 1) for traj in trajectories)
        overall_stkd = torch.cat(overalls, dim=1)
        self.overall = overall_stkd.mean(dim=1)
        self.overall_std = overall_stkd.std(dim=1)
        self.overall_sem = self.overall_std / np.sqrt(len(trajectories))


def measure_pr_ff(network: FeedforwardNetwork, pwl_prod: PointWithLabelProducer) -> PRTrajectory:
    """Measures the participation ratio through layers for the given feedforward network

    Args:
        network (FeedforwardNetwork): The feedforward network to measure pr through layers of
        pwl_prod (PointWithLabelProducer): The pointproducer to sample points from

    Returns:
        traj (PRTrajectory): the trajectory of the networks participation ratio
    """
    if not isinstance(network, FeedforwardNetwork):
        raise ValueError(f'expected network is FeedforwardNetwork, got {network} (type={type(network)})')
    if not isinstance(pwl_prod, PointWithLabelProducer):
        raise ValueError(f'expected pwl_prod is PointWithLabelProducer, got {pwl_prod} (type={type(pwl_prod)})')

    num_samples = min(pwl_prod.epoch_size, 100 * pwl_prod.output_dim)
    sample_points = torch.zeros((num_samples, network.input_dim), dtype=torch.double)
    sample_labels = torch.zeros((num_samples,), dtype=torch.long)

    pr_overall = torch.zeros(network.num_layers+1, dtype=torch.double)
    pr_by_label = torch.zeros((pwl_prod.output_dim, network.num_layers+1), dtype=torch.double)

    pwl_prod.fill(sample_points, sample_labels)

    masks_by_lbl = [sample_labels == lbl for lbl in range(pwl_prod.output_dim)]

    def on_hidacts(acts_info: FFHiddenActivations):
        hid_acts = acts_info.hidden_acts.detach()
        layer = acts_info.layer

        pr_overall[layer] = measure_pr(hid_acts)
        for lbl in range(pwl_prod.output_dim):
            pr_by_label[lbl, layer] = measure_pr(hid_acts[masks_by_lbl[lbl]])

    network(sample_points, on_hidacts)

    return PRTrajectory(overall=pr_overall, by_label=pr_by_label, layers=True)

def plot_pr_trajectory(traj: PRTrajectory, savepath: str, exist_ok: bool = False,
                       label_map: typing.Optional[typing.Dict[int, str]] = None):
    """Plots the given trajectory and saves it to the given zip archive

    Args:
        traj (PRTrajectory): The trajectory to plot
        savepath (str): Where to save the trajectory
        exist_ok (bool, optional): Defaults to False. if we should overwrite
        label_map (dict[int, str], doptional): Defaults to None. If specified,
            these are the display names for the labels. Defaults to just the
            string representation of the label. May omit any or all labels
    """
    if not isinstance(traj, PRTrajectory):
        raise ValueError(f'expected traj is PRTrajectory, got {traj} (type={type(traj)})')
    if not isinstance(savepath, str):
        raise ValueError(f'expected savepath is str, got {savepath} (type={type(savepath)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok} (type={type(exist_ok)})')

    if label_map is None and traj.by_label is not None:
        label_map = dict((lbl, str(lbl)) for lbl in range(len(traj.by_label)))
    elif traj.by_label is not None:
        if not isinstance(label_map, dict):
            raise ValueError(f'expected label_map is dict, got {label_map} (type={type(label_map)})')
        for lbl in range(len(traj.by_label)):
            if lbl not in label_map:
                label_map[lbl] = str(lbl)

    savepath_wo_ext = os.path.splitext(savepath)[0]
    if savepath == savepath_wo_ext:
        savepath += '.zip'

    if os.path.exists(savepath_wo_ext):
        raise FileExistsError(f'to save at {savepath}, {savepath_wo_ext} must be empty but it already exists')
    if not exist_ok and os.path.exists(savepath):
        raise FileExistsError(f'cannot save at {savepath} (already exists). set exist_ok=True to overwrite')

    os.makedirs(savepath_wo_ext)

    through_str = 'Layers' if traj.layers else 'Time'
    x_label = through_str
    y_label = 'Participation Ratio'

    x_vals = np.arange(traj.overall.shape[0])

    fig, axs = plt.subplots()
    axs.set_title(f'PR Through {through_str} (Global)')
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)

    axs.plot(x_vals, traj.overall.numpy())
    axs.set_xticks(x_vals)

    fig.tight_layout()
    fig.savefig(os.path.join(savepath_wo_ext, 'global.png'))
    plt.close(fig)

    fig, axs = plt.subplots()
    axs.set_title(f'PR Through {through_str} (All)')
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)

    if traj.by_label is not None:
        for lbl, y_vals in enumerate(traj.by_label):
            axs.plot(x_vals, y_vals.numpy(), '--', label=label_map[lbl], alpha=0.6)

    axs.plot(x_vals, traj.overall.numpy(), label='Overall', alpha=1)

    axs.set_xticks(x_vals)

    axs.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(savepath_wo_ext, 'all.png'))
    plt.close(fig)

    if traj.by_label is not None:
        for lbl, y_vals in enumerate(traj.by_label):
            fig, axs = plt.subplots()
            axs.set_title(f'PR Through {through_str} ({label_map[lbl]})')
            axs.set_xlabel(x_label)
            axs.set_ylabel(y_label)
            axs.plot(x_vals, y_vals.numpy())
            axs.set_xticks(x_vals)
            fig.tight_layout()
            fig.savefig(os.path.join(savepath_wo_ext, f'{lbl}.png'))
            plt.close(fig)

    traj.save(os.path.join(savepath_wo_ext, 'traj.zip'))
    if os.path.exists(savepath):
        os.remove(savepath)

    zipdir(savepath_wo_ext)

class TrajectoryWithMeta(typing.NamedTuple):
    """Describes the additional information required alongside a trajectory
    when it is plotted on a figure with other trajectories

    Attributes:
        trajectory (PRTrajectory): the actual trajectory
        label (str): a label that distinguishes this trajectory from the others
    """
    trajectory: typing.Union[PRTrajectory, AveragedPRTrajectory]
    label: str

def plot_pr_trajectories(trajectories: typing.List[TrajectoryWithMeta],
                         savepath: str, title: str, exist_ok: bool = False):
    """Plots multiple participation ratio trajectories on a single figure,
    where each trajectory must be associated with a particular label

    Arguments:
        trajectories (list[TrajectoryWithMeta]): the trajectories to plot
        savepath (str): the zip file to save the resulting figures in
        title (str): the title for the figure
        exist_ok (bool, default False): True to overwrite existing files, False not to
    """
    if not isinstance(trajectories, (list, tuple)):
        raise ValueError(f'expected trajectories is list or tuple, got {trajectories} (type={type(trajectories)})')
    if not trajectories:
        raise ValueError(f'need at least one trajectory, got empty {type(trajectories)}')
    if not isinstance(trajectories[0], TrajectoryWithMeta):
        raise ValueError(f'expected trajectories[0] is TrajectoryWithMeta, got {trajectories[0]} (type={type(trajectories[0])})')
    layers = trajectories[0].trajectory.layers
    depth = trajectories[0].trajectory.overall.shape[0]
    if not isinstance(title, str):
        raise ValueError(f'expected title is str, got {title} (type={type(title)})')
    for i, traj in enumerate(trajectories):
        if not isinstance(traj, TrajectoryWithMeta):
            raise ValueError(f'expected trajectories[{i}] is TrajectoryWithMeta, got {traj} (type={type(traj)})')
        if traj.trajectory.layers != layers:
            raise ValueError(f'trajectories[0].trajectory.layers = {layers}, trajectories[{i}].trajectory.layers = {traj.trajectory.layers}')
        _depth = traj.trajectory.overall.shape[0]
        if depth != _depth:
            raise ValueError(f'trajectories[0].trajectory.overall.shape[0] = {depth}, trajectories[{i}].trajectory.overall.shape[0] = {_depth}')

    filename, folder = mutils.process_outfile(savepath, exist_ok)
    os.makedirs(folder, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_title(title).set_fontsize(18)
    ax.set_xlabel('Layer' if layers else 'Time').set_fontsize(16)
    ax.set_ylabel('Participation Ratio').set_fontsize(16)
    ax.set_xticks([i for i in range(depth)])

    my_cmap = plt.get_cmap('Set1')
    cols = my_cmap([i for i in range(len(trajectories))])
    x_vals = np.arange(depth)
    for ind, traj_meta in enumerate(trajectories):
        traj = traj_meta.trajectory
        ax.plot(x_vals, traj.overall.numpy(), color=cols[ind], label=traj_meta.label)
    ax.legend()

    fig.savefig(os.path.join(folder, 'out.png'))
    plt.close(fig)

    if os.path.exists(filename):
        os.remove(filename)
    zipdir(folder)

def plot_avg_pr_trajectories(trajectories: typing.List[TrajectoryWithMeta],
                             savepath: str, title: str, exist_ok: bool = False):
    """Plots multiple participation ratio trajectories on a single figure,
    where each trajectory must be associated with a particular label, where
    each trajectory is actually the average of multiple trajectories

    Arguments:
        trajectories (list[TrajectoryWithMeta]): the trajectories to plot
        savepath (str): the zip file to save the resulting figures in
        title (str): the title for the figure
        exist_ok (bool, default False): True to overwrite existing files, False not to
    """
    if not isinstance(trajectories, (list, tuple)):
        raise ValueError(f'expected trajectories is list or tuple, got {trajectories} (type={type(trajectories)})')
    if not trajectories:
        raise ValueError(f'need at least one trajectory, got empty {type(trajectories)}')
    if not isinstance(trajectories[0], TrajectoryWithMeta):
        raise ValueError(f'expected trajectories[0] is TrajectoryWithMeta, got {trajectories[0]} (type={type(trajectories[0])})')
    layers = trajectories[0].trajectory.layers
    depth = trajectories[0].trajectory.overall.shape[0]
    if not isinstance(title, str):
        raise ValueError(f'expected title is str, got {title} (type={type(title)})')
    for i, traj in enumerate(trajectories):
        if not isinstance(traj, TrajectoryWithMeta):
            raise ValueError(f'expected trajectories[{i}] is TrajectoryWithMeta, got {traj} (type={type(traj)})')
        if traj.trajectory.layers != layers:
            raise ValueError(f'trajectories[0].trajectory.layers = {layers}, trajectories[{i}].trajectory.layers = {traj.trajectory.layers}')
        _depth = traj.trajectory.overall.shape[0]
        if depth != _depth:
            raise ValueError(f'trajectories[0].trajectory.overall.shape[0] = {depth}, trajectories[{i}].trajectory.overall.shape[0] = {_depth}')

    filename, folder = mutils.process_outfile(savepath, exist_ok)
    os.makedirs(folder, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_title(title).set_fontsize(18)
    ax.set_xlabel('Layer' if layers else 'Time').set_fontsize(16)
    ax.set_ylabel('Participation Ratio').set_fontsize(16)
    ax.set_xticks([i for i in range(depth)])

    my_cmap = plt.get_cmap('Set1')
    cols = my_cmap([i for i in range(len(trajectories))])
    x_vals = np.arange(depth)
    for ind, traj_meta in enumerate(trajectories):
        traj = traj_meta.trajectory
        ax.errorbar(x_vals, traj.overall.numpy(), yerr=traj.overall_sem.numpy()*1.96, color=cols[ind], label=traj_meta.label)
    ax.legend()

    fig.savefig(os.path.join(folder, 'out.png'))
    plt.close(fig)

    if os.path.exists(filename):
        os.remove(filename)
    zipdir(folder)

def digest_measure_and_plot_pr_ff(sample_points: np.ndarray, sample_labels: np.ndarray,
                        output_dim: int,
                        *all_hid_acts: typing.Tuple[np.ndarray],
                        savepath: str = None, labels: bool = False,
                        max_threads: typing.Optional[int] = 3):
    """An npmp digestable callable for measuring and plotting the participation ratio for
    a feedforward network"""

    if not isinstance(output_dim, int):
        raise ValueError(f'expected output_dim is int, got {output_dim} (type={type(output_dim)})')

    sample_points = torch.from_numpy(sample_points)
    sample_labels = torch.from_numpy(sample_labels)
    hacts_cp = []
    for hact in all_hid_acts:
        hacts_cp.append(torch.from_numpy(hact))
    all_hid_acts = hacts_cp


    num_lyrs = len(all_hid_acts)
    if labels:
        masks_by_lbl = [sample_labels == lbl for lbl in range(output_dim)]

    inqueue = myq.ZeroMQQueue.create_recieve()
    inq_serd = inqueue.serd()
    dig = npmp.NPDigestor(uuid.uuid4().hex, max_threads, 'shared.measures.participation_ratio', 'measure_pr_np')

    exp_results = len(all_hid_acts)
    if labels:
        exp_results += len(all_hid_acts) * output_dim

    for layer, hid_acts in enumerate(all_hid_acts):
        dig(hid_acts.numpy(), (layer, -1), inq_serd)
        if labels:
            for lbl in range(output_dim):
                dig(hid_acts[masks_by_lbl[lbl]].numpy(), (layer, lbl), inq_serd)

    torch_pr_overall = torch.zeros(num_lyrs, dtype=torch.double)
    if labels:
        torch_pr_by_label = torch.zeros((output_dim, num_lyrs), dtype=torch.double)

    for _ in range(exp_results):
        (layer, lbl), prval = inqueue.get()
        if lbl == -1:
            torch_pr_overall[layer] = prval
        else:
            torch_pr_by_label[lbl, layer] = prval

    traj = PRTrajectory(overall=torch_pr_overall,
                        by_label=torch_pr_by_label if labels else None,
                        layers=True)
    plot_pr_trajectory(traj, savepath, False)

def during_training_ff(savepath: str, train: bool,
                       digestor: typing.Optional[npmp.NPDigestor] = None, **kwargs):
    """Fetches the on_step/on_epoch for things like OnEpochsCaller
    that saves into the given directory.

    Args:
        savepath (str): where to save
        train (bool): true to use training data, false to use validation data
        digestor (NPDigestor, optional): if specified, used to parallelizing
        kwargs (dict): passed to plot_pr_trajectory
    """
    if not isinstance(savepath, str):
        raise ValueError(f'expected savepath is str, got {savepath} (type={type(savepath)})')
    if not isinstance(train, bool):
        raise ValueError(f'expected train is bool, got {train} (type={type(train)})')
    if digestor is not None and not isinstance(digestor, npmp.NPDigestor):
        raise ValueError(f'expected digestor is optional[NPDigestor], got {digestor} (type={type(digestor)})')

    if os.path.exists(savepath):
        raise ValueError(f'{savepath} already exists')

    def on_step(context: GenericTrainingContext, fname_hint: str):
        context.logger.info('[PR] Measuring PR Through Layers (hint: %s)', fname_hint)
        pwl = context.train_pwl if train else context.test_pwl
        outfile = os.path.join(savepath, f'pr_{fname_hint}')

        if digestor is not None:
            hacts = mutils.get_hidacts_ff(context.model, pwl).numpy()
            digestor(hacts.sample_points, hacts.sample_labels, pwl.output_dim,
                     *hacts.hid_acts,
                     savepath=outfile,
                     target_module='shared.measures.participation_ratio',
                     target_name='digest_measure_and_plot_pr_ff',
                     **kwargs)
            return

        traj = measure_pr_ff(context.model, pwl)
        plot_pr_trajectory(traj, outfile, **kwargs)

    return on_step
