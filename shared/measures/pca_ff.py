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

class PCTrajectoryFFSnapshotMatchInfo:
    """Contains the necessary information for a pc trajectory to match it with another pc
    trajectory. It is usually faster to match via a match info if you are going to match
    to the same snapshot multiple times. It is also faster to store.

    Attributes:
        num_pcs (int): the number of principal components in the snapshot that will be matched.
            We can only match trajectories with the same number of pcs
        layer_size (int): the size of the layer that will be matched
            It only makes sense to match trajectories with the same layer size
        output_dim (int): the number of output dimensions in the snapshot that will be matched.
            We can only match trajectories with the same output dimension

        mean_comps (torch.tensor uint8): has length 0.5 * num_pcs * output_dim * (output_dim - 1).
            The index tells you which pc and which 2 labels are being compared, while a
            0 tells you that lbl1 <= lbl2, 1 says lbl1 > lbl2,, where here
            "lbl1" is the mean of the first label along the given pc.

            Given pc "P", label 1 "A" and label 2 "B", the index "i" that you want is
                  P * (0.5 * output_dim * (output_dim - 1))
                + 0.5 * A * (A - 1)
                + B - A - 1
    """

    def __init__(self, num_pcs: int, layer_size: int, output_dim: int,
                 mean_comps: torch.tensor):
        if not isinstance(num_pcs, int):
            raise ValueError(f'expected num_pcs is int, got {num_pcs} (type={type(num_pcs)})')
        if not isinstance(layer_size, int):
            raise ValueError(f'expected layer_size is int, got {layer_size} (type={type(layer_size)})')
        if not isinstance(output_dim, int):
            raise ValueError(f'expected output_dim is int, got {output_dim} (type={type(output_dim)})')
        if not torch.is_tensor(mean_comps):
            raise ValueError(f'expected mean_comps is torch.tensor, got {mean_comps} (type={type(mean_comps)})')
        if mean_comps.dtype != torch.uint8:
            raise ValueError(f'expected mean_comps dtype is uint8, got {mean_comps.dtype}')
        if len(mean_comps.shape) != 1:
            raise ValueError(f'expected mean_comps is flattened, got {mean_comps.shape}')

        exp_len = PCTrajectoryFFSnapshotMatchInfo.get_expected_len(num_pcs, output_dim)
        if mean_comps.shape[0] != exp_len:
            raise ValueError(f'expected mean_comps has length {exp_len} when num_pcs={num_pcs} and output_dim={output_dim}, got {mean_comps.shape[0]}')

        self.num_pcs = num_pcs
        self.layer_size = layer_size
        self.output_dim = output_dim
        self.mean_comps = mean_comps

    @classmethod
    def create(cls, snapshot: PCTrajectoryFFSnapshot) -> 'PCTrajectoryFFSnapshotMatchInfo':
        """Creates the match information for the given snapshot"""
        num_pcs = snapshot.num_pcs
        layer_size = snapshot.layer_size
        output_dim = snapshot.projected_sample_labels.max().item() + 1

        mean_comps = torch.zeros(PCTrajectoryFFSnapshotMatchInfo.get_expected_len(num_pcs, output_dim),
                                  dtype=torch.uint8)

        means_by_label_and_pc = torch.zeros((num_pcs, output_dim), dtype=snapshot.projected_samples.dtype)

        for lbl in range(output_dim):
            mask = snapshot.projected_sample_labels == lbl
            for pc_ind in range(num_pcs):
                means_by_label_and_pc[pc_ind, lbl] = snapshot.projected_samples[mask, pc_ind].mean()

        counter = 0
        for pc_ind in range(num_pcs):
            for lbl1 in range(output_dim):
                for lbl2 in range(lbl1+1, output_dim):
                    if means_by_label_and_pc[pc_ind, lbl1] > means_by_label_and_pc[pc_ind, lbl2]:
                        mean_comps[counter] = 1
                    counter += 1

        return cls(num_pcs, layer_size, output_dim, mean_comps)

    @staticmethod
    def get_expected_len(num_pcs: int, output_dim: int) -> int:
        """Determines the expected number of comparison when there are the specified
        number of pcs and labels"""
        return (num_pcs * output_dim * (output_dim - 1)) // 2

    @staticmethod
    def get_offset_static(output_dim: int, pc_ind: int, lbl1: int, lbl2: int) -> int:
        """Gets the offset within the mean_comps array for comparing the first label to the
        second label when there are the given number of pcs and labels."""

        pc_offset = (pc_ind * output_dim * (output_dim - 1)) // 2
        label1_offset = (lbl1 * (lbl1 - 1)) // 2
        label2_offset = lbl2 - lbl1 - 1

        return pc_offset + label1_offset + label2_offset

    def get_offset(self, pc_ind: int, lbl1: int, lbl2: int):
        """Gets the offset within mean_comps for comparing lbl1 to lbl2 within the
        specified principal component vector space

        Args:
            pc (int): the index for the principal component
            lbl1 (int): the first label
            lbl2 (int): the seocnd label

        Returns:
            offset (int): the index in mean_comps to get that comparison
        """
        return PCTrajectoryFFSnapshotMatchInfo.get_offset_static(
            self.output_dim, pc_ind, lbl1, lbl2
        )

    def diff(self, match_info: 'PCTrajectoryFFSnapshotMatchInfo') -> typing.Any:
        """Produces a diff which can be sent to apply_diff to match the specified
        match information to this one.

        Args:
            match_info (PCTrajectoryFFSnapshotMatchInfo): the match info to compare with

        Returns:
            typing.Any: a result which can be sent to apply_diff
        """
        if not isinstance(match_info, PCTrajectoryFFSnapshotMatchInfo):
            raise ValueError(f'Expected match_info is PCTrajectoryFFSnapshotMatchInfo, got {match_info} (type={type(match_info)})')
        if match_info.num_pcs != self.num_pcs:
            raise ValueError(f'Cannot match because different number of pcs (I have {self.num_pcs}, arg has {match_info.num_pcs})')
        if match_info.layer_size != self.layer_size:
            raise ValueError(f'Cannot match because different layer sizes (I have {self.layer_size}, arg has {match_info.layer_size})')
        if match_info.output_dim != self.output_dim:
            raise ValueError(f'Cannot match because different output dims (I have {self.output_dim}, arg has {match_info.output_dim})')

        diff = torch.zeros((self.num_pcs,), dtype=torch.uint8)
        pc_offset = (self.output_dim * (self.output_dim - 1)) // 2
        for pc_ind in range(self.num_pcs):
            istart = pc_ind*pc_offset
            iend = istart + pc_offset
            badness = (self.mean_comps[istart:iend] != match_info.mean_comps[istart:iend]).sum()
            if badness > (pc_offset // 2):
                diff[pc_ind] = 1
        return diff

    @staticmethod
    def apply_diff(snapshot: PCTrajectoryFFSnapshot, diff: typing.Any):
        """Modifies the given snapshot in such a way that it is an acceptable result
        from the pc find trajectory but also is like the diff

        Args:
            snapshot (PCTrajectoryFFSnapshot): the snapshot to modify
            diff (typing.Any): the result from diff
        """
        snapshot.principal_vectors[diff, :] *= -1
        snapshot.projected_samples[:, diff] *= -1

    def match(self, snapshot: PCTrajectoryFFSnapshot):
        """Sets up the given snapshot to match this snapshot"""
        match_info = PCTrajectoryFFSnapshotMatchInfo.create(snapshot)
        diff = self.diff(match_info)
        PCTrajectoryFFSnapshotMatchInfo.apply_diff(snapshot, diff)

def to_trajectory(sample_labels: torch.tensor, all_hid_acts: typing.List[torch.tensor],
                  num_pcs: int) -> PCTrajectoryFF:
    """Converts the specified hidden activations to a feedforward trajectory

    Args:
        sample_labels (torch.tensor): the labels for the points sent through the model
        all_hid_acts (typing.List[torch.tensor]): the hidden activations of the network
        num_pcs (int): the number of hidden pcs to find

    Returns:
        PCTrajectoryFF: the projectory formed by the specified hidden activations
    """

    if not torch.is_tensor(sample_labels):
        raise ValueError(f'expected sample_labels is tensor, got {sample_labels} (type={type(sample_labels)})')
    if sample_labels.dtype not in (torch.uint8, torch.int, torch.long):
        raise ValueError(f'expected sample_labels is int-like but has dtype {sample_labels.dtype}')
    if len(sample_labels.shape) != 1:
        raise ValueError(f'expected sample_labels has shape [batch_size] but is {sample_labels.shape}')
    if not isinstance(all_hid_acts, (tuple, list)):
        raise ValueError(f'expected all_hid_acts is tuple or list, got {all_hid_acts} (type={type(all_hid_acts)})')

    snapshots = []
    for idx, hid_acts in enumerate(all_hid_acts):
        if not torch.is_tensor(hid_acts):
            raise ValueError(f'expected all_hid_acts[{idx}] is tensor, got {hid_acts} (type={type(hid_acts)})')
        if hid_acts.dtype not in (torch.float, torch.double):
            raise ValueError(f'expected all_hid_acts[{idx}] is float-like but has dtype {hid_acts.dtype}')
        if len(hid_acts.shape) != 2 or hid_acts.shape[0] != sample_labels.shape[0]:
            raise ValueError(f'expected all_hid_acts[{idx}].shape = [batch_size={sample_labels.shape[0]}, layer_size] but got {hid_acts.shape}')

        pc_vals, pc_vecs = get_hidden_pcs(hid_acts, num_pcs)
        projected = project_to_pcs(hid_acts, pc_vecs, out=None)
        snapshots.append(PCTrajectoryFFSnapshot(pc_vecs, pc_vals, projected, sample_labels))

    return PCTrajectoryFF(snapshots)

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

def match_snapshots(match_snap: PCTrajectoryFFSnapshot, change_snap: PCTrajectoryFFSnapshot):
    """Attempts to match the two snapshots up to reflections. The eigenvectors may be multiplied
    by an arbitrary scalar and still be an eigenvector. For the purpose of plotting it is often
    helpful to multiply by -1

    Args:
        match_snap (PCTrajectoryFFSnapshot): the snapshot we try to match
        change_snap (PCTrajectoryFFSnapshot): the snapshot we will change

    Returns:
        reflected (tuple[bool]): True if we changed the snapshot via a reflection, False if
            we did not. one for each pc
    """
    if not isinstance(match_snap, PCTrajectoryFFSnapshot):
        raise ValueError(f'expected match_snap is trajectory snapshot, got {match_snap} (type={type(match_snap)})')
    if not isinstance(change_snap, PCTrajectoryFFSnapshot):
        raise ValueError(f'expected change_snap is trajectory snapshot, got {change_snap} (type={type(change_snap)})')

    num_labels = match_snap.projected_sample_labels.max().item() + 1
    ver_num_labels = change_snap.projected_sample_labels.max().item() + 1
    if ver_num_labels != num_labels:
        raise ValueError(f'expected same number of labels between snapshots, but match has {num_labels} and change has {ver_num_labels}')

    if match_snap.num_pcs != change_snap.num_pcs:
        raise ValueError(f'expected same number of pcs between snapshots, but match has {match_snap.num_pcs} and change has {change_snap.num_pcs}')
    num_pcs = match_snap.num_pcs

    means_by_label_match = torch.zeros((num_labels, num_pcs), dtype=match_snap.projected_samples.dtype)
    for lbl in range(num_labels):
        means_by_label_match[lbl] = match_snap.projected_samples[match_snap.projected_sample_labels == lbl].mean(0)

    means_by_label_change = torch.zeros((num_labels, num_pcs), dtype=change_snap.projected_samples.dtype)
    for lbl in range(num_labels):
        means_by_label_change[lbl] = change_snap.projected_samples[change_snap.projected_sample_labels == lbl].mean(0)

    result = []
    for pc in range(match_snap.num_pcs): # pylint: disable=invalid-name
        badness = 0
        counter = 0
        for lbl1 in range(num_labels):
            for lbl2 in range(lbl1 + 1, num_labels):
                used_to_be_lt = (
                    means_by_label_match[lbl1, pc]
                    < means_by_label_match[lbl2, pc]
                )
                curr_is_lt = (
                    means_by_label_change[lbl1, pc]
                    < means_by_label_change[lbl2, pc]
                )
                counter += 1
                if used_to_be_lt != curr_is_lt:
                    badness += 1

        if badness >= (counter / 2):
            change_snap.principal_vectors[pc, :] *= -1
            change_snap.projected_samples[:, pc] *= -1
            result.append(True)
        else:
            result.append(False)

    return tuple(result)



def match_trajectories(match_traj: PCTrajectoryFF, change_traj: PCTrajectoryFF):
    """Attempts to have the two trajectories which are meant to reflect a similar system be as
    close as possible within reflections and rotations.

    Args:
        match_traj (PCTrajectoryFF): the trajectory that we're trying to match
        change_traj (PCTrajectoryFF): the trajectory that we are changing to match
    """

    if not isinstance(match_traj, PCTrajectoryFF):
        raise ValueError(f'expected match_traj is PCTrajectoryFF, got {match_traj} (type={type(match_traj)})')
    if not isinstance(change_traj, PCTrajectoryFF):
        raise ValueError(f'expected change_traj is PCTrajectoryFF, got {change_traj} (type={type(change_traj)})')
    if len(match_traj.snapshots) != len(change_traj.snapshots):
        raise ValueError(f'expected match_traj and change_traj have same # of snapshots, but match_traj has {len(match_traj.snapshots)} and change_traj has {len(change_traj.snapshots)}')

    # potential optimization is replicating the changes for the first set to all
    for idx, match_snap in enumerate(match_traj.snapshots):
        match_snapshots(match_snap, change_traj.snapshots[idx])


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
        outfile = os.path.join(savepath, f'pca_{fname_hint}')

        if digestor is not None:
            num_samples = min(200 * pwl.output_dim, pwl.epoch_size)
            hacts = mutils.get_hidacts_ff(context.model, pwl, num_samples).numpy()
            digestor(hacts.sample_points, hacts.sample_labels, *hacts.hid_acts, savepath=outfile,
                     target_module='shared.measures.pca_ff',
                     target_name='digest_find_and_plot_traj')
            return

        traj = find_trajectory(context.model, pwl, 2)
        plot_trajectory(traj, outfile)

    return on_step

