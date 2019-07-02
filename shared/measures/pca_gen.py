"""This model does principal component visualization of networks just like the other
pca_ files, except instead of labels the color is determined by an arbitrary mapping
from the out types to an (r, g, b, a) tuple.
"""
import typing
import os
import torch
import math
import numpy as np
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import shared.measures.pca as pca
import shared.typeutils as tus
import shared.measures.utils as mutils
import shared.filetools as filetools
from shared.pwl import PointWithLabelProducer
from shared.models.generic import Network
from shared.models.ff import FFHiddenActivations
from shared.models.rnn import NaturalRNN, RNNHiddenActivations

class PCTrajectoryGenSnapshot:
    """Analagous to the PCTrajectoryFFSnapshot, except this time we might have an arbitrary
    set of values for the output labels. It is assumed that the transformation from labels to
    colors is stored elsewhere.

    Attributes:
        principal_vectors (torch.tensor):
            size: [num_pcs, layer_size]
            dtype: float-like

            the first index tells you which principal component vector
            the second index tells you the weight of the corresponding feature onto the vector

        principal_values (torch.tensor):
            size: [num_pcs]
            dtype: float-like

            the first index tells you which value

        projected_samples (torch.tensor):
            size: [num_samples, num_pcs]:
            dtype: float-like

            the first index tells you which sample
            the second index tells you which principal component vector

        projected_sample_labels (torch.tensor):
            size: [num_samples, ...]
            dtype: any

            the first index tells you which sample
    """
    def __init__(self, principal_vectors: torch.tensor, principal_values: torch.tensor,
                 projected_samples: torch.tensor, projected_sample_labels: torch.tensor):
        tus.check_tensors(
            principal_vectors=(
                principal_vectors,
                (('num_pcs', None), ('layer_size', None)),
                (torch.float, torch.double)
            ),
            principal_values=(
                principal_values,
                (('num_pcs', principal_vectors.shape[0]),),
                principal_vectors.dtype
            ),
            projected_samples=(
                projected_samples,
                (('num_samples', None), ('num_pcs', principal_vectors.shape[0])),
                principal_vectors.dtype
            )
        )
        if (not projected_sample_labels.shape
                or projected_sample_labels.shape[0] != projected_samples.shape[0]):
            raise ValueError('expected projected_sample_labels.shape is (num_samples, ...) '
                             + f'but got {projected_sample_labels.shape}')

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

class PCTrajectoryGen:
    """Analagous to PCTrajectoryFF, except instead of PCTrajectoryFFSnapshot's we have
    PCTrajectoryGenSnapshot's. Stores a sequence of snapshots through either time or layers
    which all have a consistent data types and number of pcs

    Attributes:
        snapshots (list[PCTrajectoryGenSnapshot]): the snapshots through this trajectory across
            time or layers
    """

    def __init__(self, snapshots: typing.List[PCTrajectoryGenSnapshot]) -> None:
        tus.check_list(PCTrajectoryGenSnapshot, snapshots=snapshots)

        fdtype = snapshots[0].principal_vectors.dtype
        num_pcs = snapshots[0].principal_values.shape[0]
        for i, snap in enumerate(snapshots):
            if snap.principal_vectors.dtype != fdtype:
                raise ValueError(f'mismatched floating data types; for snapshots[0] is {fdtype}, '
                                 + f'for snapshots[{i}] is {snap.principal_vectors.dtype}')
            if snap.principal_values.shape[0] != num_pcs:
                raise ValueError(f'mismatched number of pcs; for snapshots[0] is {num_pcs}, '
                                 + f'for snapshots[{i}] is {snap.principal_values.shape[0]}')

        self.snapshots = snapshots

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

def to_trajectory(sample_labels: torch.tensor, all_hid_acts: typing.List[torch.tensor],
                  num_pcs: int) -> PCTrajectoryGen:
    """Converts the given labels and hidden activations to a trajectory with the specified
    number of principal components.

    Args:
        sample_labels (torch.tensor): The labels / targets for the corresponding samples
            size: [num_samples, ...]
            dtype: any

            first index is which sample

        all_hid_acts (list[torch.tensor]): the hidden activations through time or layers
            size: [num_layers]
            each element:
                size: [num_samples, layer_size]
                dtype: float-like

                the first index is which sample
                the second index is which feature

    Returns:
        The trajectory found by calculating the top num_pcs pcs and then projecting the samples
        onto those pcs. For plotting it may be helpful to use a PCTrajectoryFFMatchInfo which works
        for the Gen version as well (since it is agnostic to labels)
    """
    tus.check(sample_labels=(sample_labels, torch.tensor),
              all_hid_acts=(all_hid_acts, (list, tuple)),
              num_pcs=(num_pcs, int))
    tus.check_list(torch.tensor, all_hid_acts=all_hid_acts)

    snapshots = []
    for idx, hid_acts in enumerate(all_hid_acts):
        torch.check_tensors(
            **{f'hid_acts[{idx}]': (
                hid_acts,
                (
                    ('num_samples', sample_labels.shape[0]),
                    ('layer_size', None)
                ),
                (torch.float, torch.double)
            )}
        )
        pc_vals, pc_vecs = pca.get_hidden_pcs(hid_acts, num_pcs)
        projected = pca.project_to_pcs(hid_acts, pc_vecs, out=None)
        snapshots.append(PCTrajectoryGenSnapshot(pc_vecs, pc_vals, projected, sample_labels))

    return PCTrajectoryGen(snapshots)

FRAME_SIZE = (19.2, 10.8)
DPI = 200

def find_trajectory(model: Network, pwl_prod: PointWithLabelProducer,
                    num_pcs: int, recur_times: int = 10,
                    points_dtype: typing.Any = torch.float,
                    out_dtype: typing.Any = torch.float,
                    squeeze_to_pwl: bool = True) -> PCTrajectoryGen:
    """Finds the PC trajectory for the given network, sampling a reasonable number of
    points from the dataset. The resulting labels will simply be the shape of the output
    layer for the network.

    Args:
        model (Network): the network whose activations will be fetched
        pwl_prod (PointWithLabelProducer): the dataset to sample inputs from
        num_pcs (int): the number of principal components to project onto
        recur_times (int, optional): if the network is recurrent, this is how many timesteps
            to get the trajectory on. Default 10.
        points_dtype (torch dtype, optional): the data type for points. Default torch.float.
        out_dtype (torch dtype, optional): the data type to use. Default torch.float.
        squeeze_to_pwl (bool, optional): If the output is squeezed before filled by the point
            with label producer. Helpful if the output is a single scalar and the pwl expects
            a tensor with shape (num_samples). Default True
    """
    tus.check(
        model=(model, Network),
        pwl_prod=(pwl_prod, PointWithLabelProducer),
        num_pcs=(num_pcs, int),
        recur_times=(recur_times, int)
    )

    num_samples = min(pwl_prod.epoch_size, 2000)
    points = torch.zeros((num_samples, pwl_prod.input_dim), dtype=points_dtype)
    out = torch.zeros((num_samples, model.output_dim), dtype=out_dtype)
    pwl_prod.fill(points, out.squeeze() if squeeze_to_pwl else out)

    snapshots = []

    def on_hidacts_raw(hid_acts: torch.tensor):
        if hid_acts.shape[1] >= num_pcs:
            pc_vals, pc_vecs = pca.get_hidden_pcs(hid_acts, num_pcs)
            projected = pca.project_to_pcs(hid_acts, pc_vecs, out=None)
            snapshots.append(PCTrajectoryGenSnapshot(pc_vecs, pc_vals, projected, out))
        elif hid_acts.shape[1] > 1:
            pc_vals, pc_vecs = pca.get_hidden_pcs(hid_acts, hid_acts.shape[1])
            projected = pca.project_to_pcs(hid_acts, pc_vecs, out=None)

            pc_vecs_app = torch.zeros((num_pcs, hid_acts.shape[1]), dtype=pc_vecs.dtype)
            pc_vecs_app[:pc_vecs.shape[0]] = pc_vecs

            pc_vals_app = torch.zeros((num_pcs,), dtype=pc_vals.dtype)
            pc_vals_app[:pc_vals.shape[0]] = pc_vals

            projected_app = torch.zeros((hid_acts.shape[0], num_pcs), dtype=projected.dtype)
            projected_app[:projected.shape[0]] = projected

            snapshots.append(PCTrajectoryGenSnapshot(pc_vecs_app, pc_vals_app, projected_app, out))
        else:
            pc_vecs = torch.zeros((num_pcs, hid_acts.shape[1]), dtype=hid_acts.dtype)
            pc_vals = torch.zeros((num_pcs,), dtype=hid_acts.dtype)

            pc_vals[0] = 1
            pc_vecs[0, 0] = 1

            projected = torch.zeros((hid_acts.shape[0], num_pcs), dtype=hid_acts.dtype)
            projected[:, 0] = hid_acts.squeeze()

            snapshots.append(PCTrajectoryGenSnapshot(pc_vecs, pc_vals, projected, out))


    if isinstance(model, NaturalRNN):
        def on_hidacts_rnn(hacts: RNNHiddenActivations):
            on_hidacts_raw(hacts.hidden_acts.detach())
        model(points, recur_times, on_hidacts_rnn, 1)
    else:
        def on_hidacts_ff(facts: FFHiddenActivations):
            on_hidacts_raw(facts.hidden_acts.detach())
        model(points, on_hidacts_ff)

    return PCTrajectoryGen(snapshots)

class OutputToScalarMapping:
    """The interface for things that map outputs (sample labels) to scalars which
    can then be mapped to colors using any of matplotlibs color maps.
    """
    def __call__(self, outputs: torch.tensor,
                 scalars: typing.Optional[torch.tensor] = None) -> torch.tensor:
        """Maps the given outputs to scalars.

        Args:
            outputs (torch.tensor[num_samples, output_size]): the outputs to be mapped
            scalars (torch.tensor[num_samples]): where to store the scalar versions of the outputs
                If None, a new tensor is created with the same dtype as outputs
        Returns:
            scalars
        """
        raise NotImplementedError

class SqueezeOTSMapping(OutputToScalarMapping):
    """Simply squeezes the outputs to get the scalars. Works if the networks output is one
    dimensional
    """
    def __call__(self, outputs: torch.tensor,
                 scalars: typing.Optional[torch.tensor] = None) -> torch.tensor:
        if scalars is None:
            return outputs.clone().squeeze()
        scalars[:] = outputs.squeeze()
        return scalars

def plot_trajectory(traj: PCTrajectoryGen, filepath: str, exist_ok: bool = False,
                    alpha: float = 0.5, square: bool = True, transparent: bool = True,
                    s: int = 1, ots: OutputToScalarMapping = SqueezeOTSMapping(),
                    cmap: typing.Union[mcolors.Colormap, str] = 'cividis',
                    norm: mcolors.Normalize = mcolors.Normalize(-1, 1),
                    compress: bool = False):
    """Plots the given trajectory by storing it in the given filepath. If the output of
    the trajectory is not itself a scalar, the output to scalar mapping must be set.
    The other arguments are related to display.

    Args:
        traj (PCTrajectoryGen): The trajectory to plot. Must have at least 2 pcs
        filepath (str): Where to store the given trajectory, either a folder or a zip file.
            The file zip extension will only be used if compress is true
        exist_ok (bool, optional): If the filepath already exists, then this determines if it
            should be overwritten (True) or an error should be raised (False). Defaults to False.
        alpha (float, optional): The transparency value for each vector. Defaults to 0.5.
        square (bool, optional): If the dimensions of the space should be equal for width and
            height (such that 1 inch width and height visually corresponds to the same amount of
            distance in pc-space). Since pc space is naturally rectangular, not setting this
            can easily lead to misinterpretations. Defaults to True.
        transparent (bool, optional): Determines the background color of the saved images, where
            True is transparency and False is near-white. Defaults to True.
        s (int, optional): The size of each projected sample. Defaults to 1.
        ots (OutputToScalarMapping, optional): Maps the labels of the trajectory to samples which
            are then converted to colors using the color map. Defaults to SqueezeOTSMapping().
        cmap (str or Colormap, optional): The color map to use. Defaults to 'cividis'.
        norm (mcolors.Normalize, optional): Normalizes the scalars that are passed to the color
            map to the range 0-1. Defaults to normalizing linearly from [-1, 1] to [0, 1]
        compress (bool): if the folder should be zipped
    """
    tus.check(
        traj=(traj, PCTrajectoryGen),
        filepath=(filepath, str),
        exist_ok=(exist_ok, bool),
        alpha=(alpha, float),
        square=(square, bool),
        transparent=(transparent, bool),
        s=(s, int),
        ots=(ots, OutputToScalarMapping),
        cmap=(cmap, (str, mcolors.Colormap))
    )

    outfile, outfile_wo_ext = mutils.process_outfile(filepath, exist_ok, compress)
    if not compress and exist_ok and os.path.exists(outfile_wo_ext):
        filetools.deldir(outfile_wo_ext)
    os.makedirs(outfile_wo_ext)

    num_splots_req = traj.num_layers + 1
    closest_square: int = int(np.ceil(np.sqrt(num_splots_req)))
    num_cols: int = int(math.ceil(num_splots_req / closest_square))
    local_fig, local_axs = plt.subplots(num_cols, closest_square, squeeze=False, figsize=FRAME_SIZE)

    layer: int = 0
    for x in range(num_cols):
        for y in range(closest_square):
            if layer >= num_splots_req:
                local_axs[x][y].remove()
                continue
            elif layer >= traj.num_layers:
                lspace = np.linspace(norm.vmin, norm.vmax, 100)
                axis = local_axs[x][y]
                axis.tick_params(axis='both', which='both', bottom=False, left=False, top=False,
                                 labelbottom=False, labelleft=False)
                axis.imshow(lspace[..., np.newaxis], cmap=cmap, norm=norm, aspect=0.2)
                layer += 1
                continue
            snapshot: PCTrajectoryGenSnapshot = traj[layer]

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
            if square:
                extents_x = max_x - min_x
                extents_y = max_y - min_y
                if extents_x > extents_y:
                    upd = (extents_x - extents_y) / 2
                    min_y -= upd
                    max_y += upd
                else:
                    upd = (extents_y - extents_x) / 2
                    min_x -= upd
                    max_x += upd
            padding_x = (max_x - min_x) * .1
            padding_y = (max_y - min_y) * .1

            vis_min_x = min_x - padding_x
            vis_max_x = max_x + padding_x
            vis_min_y = min_y - padding_y
            vis_max_y = max_y + padding_y

            projected_colors = ots(projected_lbls)
            axis = local_axs[x][y]
            axis.scatter(projected[:, 0].numpy(), projected[:, 1].numpy(),
                         s=s, alpha=alpha, c=projected_colors.numpy(),
                         cmap=mcm.get_cmap(cmap), norm=norm)
            axis.set_xlim([vis_min_x, vis_max_x])
            axis.set_ylim([vis_min_y, vis_max_y])
            axis.tick_params(axis='both', which='both', bottom=False, left=False, top=False,
                             labelbottom=False, labelleft=False)
            layer += 1

    local_path = os.path.join(outfile_wo_ext, 'local.png')
    local_fig.tight_layout()
    local_fig.savefig(local_path, transparent=transparent, DPI=DPI)

    np.savez(os.path.join(outfile_wo_ext, 'principal_vectors.npz'),
             *[snapshot.principal_vectors for snapshot in traj])
    np.savez(os.path.join(outfile_wo_ext, 'principal_values.npz'),
             *[snapshot.principal_values for snapshot in traj])
    np.savez(os.path.join(outfile_wo_ext, 'projected_samples.npz'),
             *[snapshot.projected_samples for snapshot in traj])
    np.savez(os.path.join(outfile_wo_ext, 'projected_sample_labels.npz'),
             *[snapshot.projected_sample_labels for snapshot in traj])

    if compress:
        if os.path.exists(outfile):
            os.remove(outfile)

        filetools.zipdir(outfile_wo_ext)
