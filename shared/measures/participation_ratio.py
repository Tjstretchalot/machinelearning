"""Produces plots of participation ratios, either through time or through
layers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import typing
import os
import shared.measures.pca as pca
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations
from shared.pwl import PointWithLabelProducer
from shared.filetools import zipdir
from shared.trainer import GenericTrainingContext
import shared.npmp as npmp
import shared.measures.utils as mutils

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

    eigs, _ = pca.get_hidden_pcs(hidden_acts, None)

    result = torch.pow(torch.sum(eigs), 2) / torch.sum(torch.pow(eigs, 2))
    result = result.item()

    if not isinstance(result, float):
        raise ValueError(f'expected participation_ratio is float, got {result} (type={type(result)})')
    if result < 1 - 1e-8:
        raise ValueError(f'PR should always be >= 1 (got {result})')
    if result > eigs.shape[0] + 1e-8:
        raise ValueError(f'PR should always be <= M where M = eigs.shape[0] = {eigs.shape[0]} (got {result})')

    return result

class PRTrajectory(typing.NamedTuple):
    """Describes the trajectory of participation ratio through time or layers

    Attributes:
        overall (torch.tensor[number layers / recurrent times + 1])
        by_label (list[torch.tensor[number layers / recurrent_times + 1]])
        layers (bool): true if this is a through-layers plot, false if through-time
    """
    overall: torch.tensor
    by_label: typing.List[torch.tensor]
    layers: bool

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

    if label_map is None:
        label_map = dict((lbl, str(lbl)) for lbl in range(len(traj.by_label)))
    else:
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

    for lbl, y_vals in enumerate(traj.by_label):
        axs.plot(x_vals, y_vals.numpy(), '--', label=label_map[lbl], alpha=0.6)

    axs.plot(x_vals, traj.overall.numpy(), label='Overall', alpha=1)

    axs.set_xticks(x_vals)

    axs.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(savepath_wo_ext, 'all.png'))
    plt.close(fig)

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

    if os.path.exists(savepath):
        os.remove(savepath)

    zipdir(savepath_wo_ext)

def digest_measure_and_plot_pr_ff(sample_points: np.ndarray, sample_labels: np.ndarray,
                        *all_hid_acts: typing.Tuple[np.ndarray],
                        savepath: str = None):
    """An npmp digestable callable for measuring and plotting the participation ratio for
    a feedforward network"""

    sample_points = torch.from_numpy(sample_points)
    sample_labels = torch.from_numpy(sample_labels)
    hacts_cp = []
    for hact in all_hid_acts:
        hacts_cp.append(torch.from_numpy(hact))
    all_hid_acts = hacts_cp

    num_lyrs = len(all_hid_acts)
    output_dim = sample_labels.max().item()
    if not isinstance(output_dim, int):
        raise ValueError(f'expected output_dim is int, got {output_dim} (type={type(output_dim)})')

    pr_overall = torch.zeros(num_lyrs, dtype=torch.double)
    pr_by_label = torch.zeros((output_dim, num_lyrs), dtype=torch.double)

    masks_by_lbl = [sample_labels == lbl for lbl in range(output_dim)]

    for layer, hid_acts in enumerate(all_hid_acts):
        pr_overall[layer] = measure_pr(hid_acts)
        for lbl in range(output_dim):
            pr_by_label[lbl, layer] = measure_pr(hid_acts[masks_by_lbl[lbl]])

    traj = PRTrajectory(overall=pr_overall, by_label=pr_by_label, layers=True)
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
            digestor(hacts.sample_points, hacts.sample_labels, *hacts.hid_acts,
                     savepath=outfile,
                     target_module='shared.measures.participation_ratio',
                     target_name='digest_measure_and_plot_pr_ff',
                     **kwargs)
            return

        traj = measure_pr_ff(context.model, pwl)
        plot_pr_trajectory(traj, outfile, **kwargs)

    return on_step
