"""Does PCA specifically for a deep2-style network, where the output layer
consists of 1-hot encodings for the preferred action to make. For plotting,
the preferred action is shown by the marker shape and the expected reward from
that action is shown by the marker color.
"""
import typing
import os
import math

import torch
import numpy as np
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import shared.measures.pca_gen as pca_gen
import shared.measures.utils as mutils
import shared.typeutils as tus
import shared.filetools as filetools

FRAME_SIZE = (19.2, 10.8)
DPI = 200

def plot_trajectory(traj: pca_gen.PCTrajectoryGen, filepath: str, exist_ok: bool = False,
                    markers: typing.List[str] = ('<', '>', '^', 'v'),
                    cmap: typing.Union[mcolors.Colormap, str] = 'cividis',
                    norm: mcolors.Normalize = mcolors.Normalize(-1, 1),
                    transparent: bool = False):
    """Plots the given trajectory (from a deep2-style network) to the given
    folder.

    Arguments:
        traj (PCTrajectoryGen): the trajectory to plot
        filepath (str): where to save the output, should be a folder
        exist_ok (bool): False to error if the filepath exists, True to delete it
            if it already exists
        markers (list[str]): the marker corresponding to each preferred action
        cmap (str or Colormap, optional): The color map to use. Defaults to 'cividis'.
        norm (mcolors.Normalize, optional): Normalizes the scalars that are passed to the color
            map to the range 0-1. Defaults to normalizing linearly from [-1, 1] to [0, 1]
        transparent (bool): True for a transparent background, False for a white one
    """
    tus.check(
        traj=(traj, pca_gen.PCTrajectoryGen),
        filepath=(filepath, str),
        exist_ok=(exist_ok, bool),
    )
    tus.check_list(str, markers=markers)

    ots = pca_gen.MaxOTSMapping()
    s = 12
    alpha = 0.8


    outfile_wo_ext = mutils.process_outfile(filepath, exist_ok, False)[1]
    if exist_ok and os.path.exists(outfile_wo_ext):
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
            snapshot: pca_gen.PCTrajectoryGenSnapshot = traj[layer]

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

            markers_selected = projected_lbls.max(dim=1)[1]
            axis = local_axs[x][y]
            for marker_ind, marker in enumerate(markers):
                marker_projected = projected[markers_selected == marker_ind]
                marker_projected_lbls = projected_lbls[markers_selected == marker_ind]
                projected_colors = ots(marker_projected_lbls)
                axis.scatter(marker_projected[:, 0].numpy(), marker_projected[:, 1].numpy(),
                             s=s, alpha=alpha, c=projected_colors.numpy(),
                             cmap=mcm.get_cmap(cmap), norm=norm, marker=marker)

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
