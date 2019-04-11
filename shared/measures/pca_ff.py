"""Principal components analysis.

This is the analogue for feed forward networks. The idea is largely the same, but
is more complicated for feed-forward networks compared to recurrent networks which
have the same size at each timestep.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations
from shared.pwl import PointWithLabelProducer
from shared.measures.pca import get_hidden_pcs, project_to_pcs, plot_snapshot
from shared.trainer import GenericTrainingContext
import shared.npmp as npmp
import shared.measures.utils as mutils

import typing
import os
import math
import shutil

class PCTrajectoryFFSnapshot:
    """Describes the principal components of a network at a particular layer. Unlike
    in the recurrent space we do not necessarily have the same size between snapshots
    so we cannot trivially vectorize these. Should be treated as read-only.

    Attributes:
        principal_vectors (torch.tensor):
            size: [num_pcs, layer_size]
            dtype: float-like

            the first index tells you which vector
            the second index tells you which hidden activation

        principal_values (torch.tensor):
            size: [num_pcs]
            dtype: float-like

            the first index tells you which value

        projected_samples (torch.tensor):
            size: [num_samples, num_pcs]
            dtype: float-like

            the first index tells you which sample
            the second index tells you which pc

        projected_sample_labels (torch.tensor):
            size: [num_samples]
            dtype: int-like

            the first index tells you which sample
    """

    def __init__(self, principal_vectors: torch.tensor, principal_values: torch.tensor,
                 projected_samples: torch.tensor, projected_sample_labels: torch.tensor):
        if not torch.is_tensor(principal_vectors):
            raise ValueError(f'expected principal_vectors is tensor, got {principal_vectors} (type={type(principal_vectors)})')
        if principal_vectors.dtype not in (torch.float, torch.double):
            raise ValueError(f'expected principal_vectors is float-like, dtype is {principal_vectors.dtype}')
        if len(principal_vectors.shape) != 2:
            raise ValueError(f'expected principal_vectors has shape (num_pcs, layer_size) but has shape {principal_vectors.shape}')
        if not torch.is_tensor(principal_values):
            raise ValueError(f'expected principal_values is tensor, got {principal_values} (type={type(principal_values)})')
        if principal_values.dtype not in (torch.float, torch.double):
            raise ValueError(f'expected principal_values is float-like, dtype is {principal_values.dtype}')
        if len(principal_values.shape) != 1:
            raise ValueError(f'expected principal_values has shape (num_pcs) but has shape {principal_values.shape}')
        if principal_values.shape[0] != principal_vectors.shape[0]:
            raise ValueError(f'expected principal_vectors has shape (num_pcs, layer_size) and principal_values has shape (num_pcs) but principal_vectors shape = {principal_vectors.shape} and principal_values shape = {principal_values.shape} (mismatch on dim 0)')
        if principal_values.dtype != principal_vectors.dtype:
            raise ValueError(f'principal_vectors.dtype = {principal_vectors.dtype}, principal_values.dtype={principal_values.dtype} - mismatch')
        if not torch.is_tensor(projected_samples):
            raise ValueError(f'expected projected_samples is tensor, got {projected_samples} (type={type(projected_samples)})')
        if projected_samples.dtype not in (torch.float, torch.double):
            raise ValueError(f'expected projected_samples is float-like, dtype is {projected_samples.dtype}')
        if len(projected_samples.shape) != 2:
            raise ValueError(f'expected projected_samples has shape (num_samples, num_pcs) but has shape {projected_samples.shape}')
        if projected_samples.shape[1] != principal_values.shape[0]:
            raise ValueError(f'expected principal_values has shape (num_pcs) and projected_samples has shape (num_samples, num_pcs) but principal_values.shape = {principal_values.shape} and projected_ssamples.shape = {projected_samples.shape}')
        if projected_samples.dtype != principal_values.dtype:
            raise ValueError(f'principal_values.dtype = {principal_values.dtype}, projected_samples.dtype={projected_samples.dtype} - mismatch')
        if not torch.is_tensor(projected_sample_labels):
            raise ValueError(f'expected projected_sample_labels is tensor, got {projected_sample_labels} (type={type(projected_sample_labels)})')
        if projected_sample_labels.dtype not in (torch.uint8, torch.int, torch.long):
            raise ValueError(f'expected projected_sample_labels is int-like, dtype is {projected_sample_labels.dtype}')
        if len(projected_sample_labels.shape) != 1:
            raise ValueError(f'expected projected_sample_labels has shape (num_samples), but has shape {projected_sample_labels.shape}')
        if projected_sample_labels.shape[0] != projected_samples.shape[0]:
            raise ValueError(f'expected projected_samples has shape (num_samples, num_pcs) and projected_sample_labels has shape (num_samples) but projected_samples.shape={projected_samples.shape} and projected_sample_labels.shape={projected_sample_labels.shape} (mismatch on dim 0)')

        self.principal_vectors = principal_vectors
        self.principal_values = principal_values
        self.projected_samples = projected_samples
        self.projected_sample_labels = projected_sample_labels

    @property
    def num_pcs(self):
        """Gets the number of principal components in this snapshot"""
        return self.principal_values.shape[0]

    @property
    def layer_size(self):
        """Gets the intrinsic layer dimensionality"""
        return self.principal_vectors.shape[1]

    @property
    def num_samples(self):
        """Get the number of samples projected"""
        return self.projected_samples.shape[0]

class PCTrajectoryFF:
    """Describes a pc trajectory for a feed-forward network. This passes through to the
    underlying snapshots tuple.

    Attributes:
        snapshots (tuple[PCTrajectoryFFSnapshot]) - layer snaphots, where index 0 is the
            input space and index -1 is the output space
    """

    def __init__(self, snapshots: typing.Tuple[PCTrajectoryFFSnapshot]):
        self.snapshots = tuple(snapshots)

        if not self.snapshots:
            raise ValueError(f'must have at least one snapshot (got {snapshots} -> {self.snapshots})')

        num_pcs = None
        for layer, snapshot in enumerate(self.snapshots):
            snapshot: PCTrajectoryFFSnapshot
            if num_pcs is None:
                num_pcs = snapshot.num_pcs
            elif num_pcs != snapshot.num_pcs:
                raise ValueError(f'snapshots should all have same # pcs, but snapshots[0].num_pcs={num_pcs} and snapshots[{layer}].num_pcs={snapshot.num_pcs}')

    @property
    def num_layers(self):
        """Gets the number of layers in this trajectory"""
        return len(self.snapshots)

    @property
    def input_dim(self):
        """Returns the input space embedded space dimensionality"""
        return self.snapshots[0].layer_size

    @property
    def output_dim(self):
        """Returns the output space embedded space dimensionality"""
        return self.snapshots[-1].layer_size

    @property
    def num_pcs(self):
        """Returns the number of pcs in this trajectory"""
        return self.snapshots[0].num_pcs

    def __len__(self):
        return len(self.snapshots)

    def __iter__(self):
        return iter(self.snapshots)

    def __getitem__(self, i):
        return self.snapshots[i]

def find_trajectory(model: FeedforwardNetwork, pwl_prod: PointWithLabelProducer,
                    num_pcs: int) -> PCTrajectoryFF:
    """Finds the pc trajectory for the given feed-forward model. Gets only the top
    num_pcs principal components

    Args:
        model (FeedforwardNetwork): the model to get the pc trajectory of
        pwl_prod (PointWithLabelProducer): the points to pass through
        num_pcs (int): the number of pcs to get

    Returns:
        PCTrajectoryFF: the pc trajectory of the network for the sampled points
    """
    if not isinstance(model, FeedforwardNetwork):
        raise ValueError(f'expected model is FeedforwardNetwork, got {model} (type={type(model)})')
    if not isinstance(pwl_prod, PointWithLabelProducer):
        raise ValueError(f'expected pwl_prod is PointWithLabelProducer, got {pwl_prod} (type={type(pwl_prod)})')
    if not isinstance(num_pcs, int):
        raise ValueError(f'expected num_pcs is int, got {num_pcs}')
    if num_pcs <= 0:
        raise ValueError(f'expected num_pcs is positive, got {num_pcs}')

    num_samples = min(pwl_prod.epoch_size, 200 * pwl_prod.output_dim)
    sample_points = torch.zeros((num_samples, model.input_dim), dtype=torch.double)
    sample_labels = torch.zeros((num_samples,), dtype=torch.long)
    snapshots = [] # we will fill with PCTrajectoryFFSnapshot's

    pwl_prod.fill(sample_points, sample_labels)

    def on_hidacts(acts_info: FFHiddenActivations):
        hid_acts = acts_info.hidden_acts.detach()

        pc_vals, pc_vecs = get_hidden_pcs(hid_acts, num_pcs)
        projected = project_to_pcs(hid_acts, pc_vecs, out=None)
        snapshots.append(PCTrajectoryFFSnapshot(pc_vecs, pc_vals, projected, sample_labels))

    model(sample_points, on_hidacts)

    return PCTrajectoryFF(snapshots)

def plot_trajectory(traj: PCTrajectoryFF, filepath: str, exist_ok: bool = False, alpha=0.5):
    """Plots the given trajectory and saves it to the given filepath.

    Args:
        traj (PCTrajectoryFF): the trajectory to plot
        filepath (str): where to save the images and data. should have extension 'zip' or no extension
        exist_ok (bool, default false): if true we will overwrite existing files rather than error
        alpha (float): alpha for the points
    """
    if not isinstance(traj, PCTrajectoryFF):
        raise ValueError(f'expected traj to be PCTrajectoryFF, got {traj} (type={type(traj)})')
    if not isinstance(filepath, str):
        raise ValueError(f'expected filepath to be str, got {filepath} (type={type(filepath)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok to be bool, got {exist_ok} (type={type(exist_ok)})')
    if not isinstance(alpha, float):
        raise ValueError(f'expected alpha is float, got {alpha} (type={type(alpha)})')
    if alpha < 0 or alpha > 1:
        raise ValueError(f'expected alpha in [0, 1], got {alpha}')

    filepath_wo_ext = os.path.splitext(filepath)[0]
    if filepath_wo_ext == filepath:
        filepath += '.zip'

    if os.path.exists(filepath_wo_ext):
        raise FileExistsError(f'for filepath {filepath} we require {filepath_wo_ext} is available (already exists)')

    if not exist_ok and os.path.exists(filepath):
        raise FileExistsError(f'filepath {filepath} already exists (use exist_ok=True to overwrite)')

    os.makedirs(filepath_wo_ext)

    closest_square: int = int(np.ceil(np.sqrt(traj.num_layers)))
    num_cols: int = int(math.ceil(traj.num_layers / closest_square))

    local_fig, local_axs = plt.subplots(num_cols, closest_square, squeeze=False)

    layer: int = 0
    for x in range(num_cols):
        for y in range(closest_square):
            if layer >= traj.num_layers:
                local_axs[x][y].remove()
                continue
            snapshot: PCTrajectoryFFSnapshot = traj[layer]

            projected = snapshot.projected_samples
            projected_lbls = snapshot.projected_sample_labels

            min_x, min_y, max_x, max_y = (torch.min(projected[:, 0]), torch.min(projected[:, 1]),
                                          torch.max(projected[:, 0]), torch.max(projected[:, 1]))
            min_x, min_y, max_x, max_y = min_x.item(), min_y.item(), max_x.item(), max_y.item()

            if max_x - min_x < 1e-3:
                min_x -= 5e-4
                max_x += 5e-4
            if max_y - min_y < 1e-3:
                min_y -= 5e-4
                max_y += 5e-4
            padding_x = (max_x - min_x) * .1
            padding_y = (max_y - min_y) * .1

            plot_snapshot(local_axs[x][y], projected, projected_lbls, min_x - padding_x, max_x + padding_x,
                          min_y - padding_y, max_y + padding_y, alpha=alpha)
            layer += 1

    local_path = os.path.join(filepath_wo_ext, 'local.png')
    local_fig.tight_layout()
    local_fig.savefig(local_path, transparent=True)

    np.savez(os.path.join(filepath_wo_ext, 'principal_vectors.npz'), *[snapshot.principal_vectors for snapshot in traj])
    np.savez(os.path.join(filepath_wo_ext, 'principal_values.npz'), *[snapshot.principal_values for snapshot in traj])
    np.savez(os.path.join(filepath_wo_ext, 'projected_samples.npz'), *[snapshot.projected_samples for snapshot in traj])
    np.savez(os.path.join(filepath_wo_ext, 'projected_sample_labels.npz'), *[snapshot.projected_sample_labels for snapshot in traj])

    if os.path.exists(filepath):
        os.remove(filepath)

    cwd = os.getcwd()
    shutil.make_archive(filepath_wo_ext, 'zip', filepath_wo_ext)
    os.chdir(cwd)
    shutil.rmtree(filepath_wo_ext)
    os.chdir(cwd)

def digest_find_and_plot_traj(sample_points: np.ndarray, sample_labels: np.ndarray, # pylint: disable=unused-argument
                              *all_hid_acts: typing.Tuple[np.ndarray],
                              savepath: str = None, **kwargs):
    """Digestor friendly way to find the pc trajectory and then plot it at the given
    savepath for the given sample points, labels, and hidden activations"""

    sample_points = torch.from_numpy(sample_points)
    sample_labels = torch.from_numpy(sample_labels)
    hacts_cp = []
    for hact in all_hid_acts:
        hacts_cp.append(torch.from_numpy(hact))
    all_hid_acts = hacts_cp

    snapshots = []

    for hid_acts in all_hid_acts:
        pc_vals, pc_vecs = get_hidden_pcs(hid_acts, 2)
        projected = project_to_pcs(hid_acts, pc_vecs, out=None)
        snapshots.append(PCTrajectoryFFSnapshot(pc_vecs, pc_vals, projected, sample_labels))

    traj = PCTrajectoryFF(snapshots)
    plot_trajectory(traj, savepath, False, **kwargs)


def during_training(savepath: str, train: bool, digestor: typing.Optional[npmp.NPDigestor] = None):
    """Fetches the on_step/on_epoch for things like OnEpochsCaller
    that saves into the given directory.

    Args:
        savepath (str): where to save
        train (bool): true to use training data, false to use validation data
        digestor (NPDigestor, optional): if specified, used for multiprocessing
    """
    if not isinstance(savepath, str):
        raise ValueError(f'expected savepath is str, got {savepath} (type={type(savepath)})')
    if not isinstance(train, bool):
        raise ValueError(f'expected train is bool, got {train} (type={type(train)})')
    if digestor is not None and not isinstance(digestor, npmp.NPDigestor):
        raise ValueError(f'expected digestor is NPDigestor, got {digestor} (type={type(digestor)})')

    if os.path.exists(savepath):
        raise ValueError(f'{savepath} already exists')

    def on_step(context: GenericTrainingContext, fname_hint: str):
        context.logger.info('[PCA_FF] Measuring PCA Through Layers (hint: %s)', fname_hint)
        pwl = context.train_pwl if train else context.test_pwl

        if digestor is not None:
            num_samples = min(200 * pwl.output_dim, pwl.epoch_size)
            hacts = mutils.get_hidacts_ff(context.model, pwl, num_samples).numpy()
            digestor(hacts.sample_points, hacts.sample_labels, *hacts.hid_acts, savepath=savepath,
                     target_module='shared.measures.pca_ff',
                     target_name='digest_find_and_plot_traj')
            return

        traj = find_trajectory(context.model, pwl, 2)
        plot_trajectory(traj, os.path.join(savepath, f'pca_{fname_hint}'))

    return on_step

