"""Principal component analysis.

This is a measure for the output space of a recurrent neural network. It finds
the best two vectors for explaining the variance in the output space when
solving a particular problem, then project the activations onto those vectors
so that they can be displayed two-dimensionally.
"""

import torch
import numpy as np
import scipy.linalg
import matplotlib as mpl
try:
    import skcuda.linalg   # pylint: disable=import-error
    import pycuda.gpuarray # pylint: disable=import-error
    _have_cuda = True
except:
    _have_cuda = False

import math

from shared.models.rnn import NaturalRNN, RNNHiddenActivations
from shared.pwl import PointWithLabelProducer

import matplotlib.pyplot as plt
import os
import shutil
import time
import warnings
import typing
import operator
from functools import reduce

class PCTrajectory:
    """Describes the principal components of a recurrent network through time

    Attributes:
        principal_vectors (torch.tensor):
            The principal component vectors. Three-dimensional array:
                1. the first index tells you which time point
                2. the second index tells you which vector
                3. the third index tells you which hidden activation

        principal_values (torch.tensor):
            The relative importance of each of the principal component vectors. This is proportional
            to the amount of variability explained by the principal vector with the same index.
            This array is in sorted descending order.

            Two-dimensional array:
                1. the first index tells you which time point
                2. the second index tells you which vector

        projected_samples (torch.tensor):
            The samples projected along the principal component vectors at various
            time points. Three dimensional array:
                1. the first index tells you which time point
                2. the second index tells you which sample
                3. the third index tells you which principal vector

        projected_sample_labels (torch.tensor):
            The labels for the samples. One-dimensional array:
                1. the first index tells you which sample

        recurrent_space_size (int):
            The number of hidden neurons, which correponds with the natural dimensionality of
            the hidden activation space
    """

    def __init__(self, principal_vectors, principal_values, projected_samples,
                 projected_sample_labels, recurrent_space_size):
        self.principal_vectors = principal_vectors
        self.principal_values = principal_values
        self.projected_samples = projected_samples
        self.projected_sample_labels = projected_sample_labels
        self.recurrent_space_size = recurrent_space_size

    def get_num_pcs(self):
        """Gets the number of principal component vectors considered in this trajectory

        Returns:
            (int) the number of principal components considered
        """
        return self.principal_vectors.shape[1]

    def get_num_timesteps(self):
        """Gets the number of timesteps that are considered in this trajectory

        Returns:
            (int) the number of timesteps in this trajectory
        """
        return self.principal_vectors.shape[0]

    def get_num_samples(self):
        """Gets the number of samples that are available in this trajectory

        Returns:
            (int) the number of samples in this trajectory
        """
        return self.projected_sample_labels.shape[0]

    def get_recurrent_space_size(self):
        """Returns the number of recurrent nodes. Each sample corresponds with
        the activation of every recurrent node, thus the maximum dimensionality
        of the recurrent nodes is the number of hidden activations and thus the
        number of recurrent nodes.

        Returns:
            (int) the dimensionality of the sample space
        """
        return self.recurrent_space_size

    def get_pc_vec(self, recur_time, vector_ind):
        """Gets the principal component vector at the given point in time.

        Args:
            recur_time (int): After how many recurrent steps. 0 is after input is first displayed
            vector_ind (int): Which vector. The 0th explains the most variance, then 1st, ect

        Returns:
            A numpy array
        """
        return self.principal_vectors[recur_time, vector_ind]

    def get_pc_val(self, recur_time, vector_ind):
        """Gets the the magnitude corresponding with the given principal component vector at
        the given point in time

        Args:
            recur_time (int): After how many recurrent steps
            vector_ind (int): Which vector.


        """
        return self.principal_values[recur_time, vector_ind]

    def get_projected_sample(self, recur_time, sample_ind):
        """Gets the given sample projected onto the principal components vector.

        Args:
            recur_time (int): The timestep to project the sample onto
            sample_ind (int): The index for the stored sample to consider

        Returns:
            (np.array) A vector that is get_num_pcs() long, where the first value is
            the projection along the first principal component vector, the second value
            the second vector, etc.

            (int) the label that the sample has
        """
        return (self.projected_samples[recur_time, sample_ind],
                self.projected_sample_labels[recur_time, sample_ind])

def get_hidden_pcs(hidden_acts: torch.tensor, num_pcs: typing.Optional[int], vecs=True,
                   gpu_accel=True):
    """Fetches the principal component values and vectors corresponding with the given hidden
    activations.

    Args:
        hidden_acts (torch.tensor[num_samples x num_hidden]):
            The hidden activations across the samples of interest. Two dimensional array:
                1. The first index tells you which sample
                2. The second index tells you which hidden node
        num_pcs (int): The number of pcs to return (None for all)
        vecs (bool): if True, eigenvectors are returned. If false, only eigenvalues are returned
        gpu_accel (bool): if False gpu acceleration is never used, if True it is used if it makes
            sense
    Returns:
        eigs (torch.tensor[num_pcs]): the relative importance in sorted descending order for the pc vectors
        eig_vecs (torch.tensor[num_pcs x num_hidden]): the principal component vectors
    """
    if not torch.is_tensor(hidden_acts):
        raise ValueError(f'expected hidden_acts is torch.tensor, got {hidden_acts}')
    if len(hidden_acts.shape) != 2:
        raise ValueError(f'expected hidden_acts.shape is (num_samples, num_hidden), got {hidden_acts}')
    if num_pcs is not None and not isinstance(num_pcs, int):
        raise ValueError(f'expected num_pcs is int, got {num_pcs}')
    if num_pcs is not None and num_pcs < 1:
        raise ValueError(f'expected num_pcs is positive, got {num_pcs}')

    hidden_acts_np = hidden_acts.numpy().copy()
    hidden_acts_np -= np.mean(hidden_acts_np, axis=0)
    cov = np.cov(hidden_acts_np.T)
    if not vecs:
        eig = scipy.linalg.eigvals(cov)
        eig = np.real(eig)
        np.sort(eig)
        return torch.tensor(eig, dtype=hidden_acts.dtype)

    if _have_cuda and gpu_accel and (reduce(operator.mul, cov.shape) > 10000):
        # https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.linalg.eig.html
        gpu_cov = pycuda.gpuarray.to_gpu(cov)
        vr_gpu, w_gpu = skcuda.linalg.eig(gpu_cov, 'N', 'V', 'F')
        eig = w_gpu.get()
        eig_vecs = vr_gpu.get()
    else:
        eig, eig_vecs = scipy.linalg.eig(cov)
    eig = np.real(eig)
    ind = np.argsort(np.abs(eig))[::-1]
    eig_vecs = np.real(eig_vecs[:, ind])
    eig = eig[ind]

    if num_pcs is not None:
        eig_vecs = eig_vecs[:, 0:num_pcs]
        eig = eig[0:num_pcs]

    eig_vecs = eig_vecs.transpose()
    reig, reig_vecs = torch.tensor(eig, dtype=hidden_acts.dtype), torch.tensor(eig_vecs, dtype=hidden_acts.dtype)

    if not torch.is_tensor(reig):
        raise ValueError(f'expected reig is torch.tensor, got {reig}')
    if len(reig.shape) != 1:
        raise ValueError(f'expected reig.shape=(num_pcs), got {reig.shape}')
    if num_pcs is not None and reig.shape[0] != num_pcs:
        raise ValueError(f'expected reig.shape=(num_pcs), got ({reig.shape[0]}) (num_pcs={num_pcs})')
    if not torch.is_tensor(reig_vecs):
        raise ValueError(f'expected reig_vecs is torch.tensor, got {reig_vecs}')
    if len(reig_vecs.shape) != 2:
        raise ValueError(f'expected reig_vecs.shape=(num_pcs, num_hidden), got {reig_vecs.shape}')
    if (num_pcs is not None and reig_vecs.shape[0] != num_pcs) or reig_vecs.shape[1] != hidden_acts.shape[1]:
        raise ValueError(f'expected reig_vecs.shape=(num_pcs, num_hidden), got {reig_vecs.shape} (num_pcs={num_pcs}, num_hidden={hidden_acts.shape[1]})')

    last = float('inf')
    for _reig in reig:
        if abs(_reig) >= abs(last) + 1e-8: # small epsilon required
            raise ValueError(f'expected reig to be sorted, but got {reig} (_reig={_reig}, last={last})')
        last = abs(float(_reig))
    return reig, reig_vecs

def project_to_pcs(points, pcs, out) -> torch.tensor:
    """Projects the given points to the given principal component vectors

    Args:
        points (torch.tensor[num_points x points_dim]): The points to project
        pcs (torch.tensor[num_pcs x points_dim]): The principal component vectors to project onto
        out (torch.tensor[num_points x num_pcs]): (the result): the projection of the points along
            the pcs, overwritten. Optional.

    Returns:
        out (torch.tensor[num_points x num_pcs]): the projection of the points salong the pcs.
            If out is provided, this is just out.
    """
    if not torch.is_tensor(points):
        raise ValueError(f'expected points is torch.tensor, got {points}')
    if not torch.is_tensor(pcs):
        raise ValueError(f'expected pcs is torch.tensor, got {pcs}')
    if len(points.shape) != 2:
        raise ValueError(f'expected points.shape is (num_points, points_dim) but got {points.shape}')
    if len(pcs.shape) != 2:
        raise ValueError(f'expected pcs.shape is (num_pcs, points_dim) but got {pcs.shape}')
    if points.shape[1] != pcs.shape[1]:
        raise ValueError(f'expected points.shape is (num_points, points_dim) and pcs.shape is (num_pcs, points_dim) but points.shape={points.shape} and pcs.shape={pcs.shape} (dont match on dim 1)')
    if out is not None:
        if not torch.is_tensor(out):
            raise ValueError(f'expected out is torch.tensor, got {out}')
        if len(out.shape) != 2:
            raise ValueError(f'expected out.shape is (num_points, num_pcs) but got {out.shape}')
        if out.shape[0] != points.shape[0]:
            raise ValueError(f'expected points.shape is (num_points, points_dim) and out.shape is (num_points, num_pcs) but points.shape={points.shape} and out.shape={out.shape} (dont match on dim 0)')
        if out.shape[1] != pcs.shape[0]:
            raise ValueError(f'expected pcs.shape is (num_pcs, points_dim) and out.shape is (num_points, num_pcs) but pcs.shape={pcs.shape} and out.shape={out.shape} (num_pcs not the same)')
    pca_proj = np.dot(points.numpy(), pcs.transpose(0, 1).numpy())
    if out is not None:
        out[:] = torch.tensor(pca_proj)
        return out
    return torch.tensor(pca_proj)

def find_trajectory(model: NaturalRNN, pwl_prod: PointWithLabelProducer,
                    duration: int, num_pcs: int) -> PCTrajectory:
    """Finds the trajectory of the given model using the given point with label producer. Goes
    through the entire epoch for the point with label producer.

    Args:
        model (Natural): The underlying model whose trajectories are being considered
        pwl_prod (PointWithLabelProducer): The producer for the samples. The entire epoch
            is gone through
        duration (int): How many timesteps to go through
        num_pcs (int): The number of principal vectors to find
    """

    num_samples = min(pwl_prod.epoch_size, 100 * pwl_prod.output_dim)
    sample_points = torch.zeros((num_samples, model.input_dim), dtype=torch.double)
    sample_labels = torch.zeros((num_samples,), dtype=torch.long)
    hid_acts = torch.zeros((duration+1, num_samples, model.hidden_dim), dtype=torch.double)
    hid_pc_vals = torch.zeros((duration+1, num_pcs), dtype=torch.double)
    hid_pc_vecs = torch.zeros((duration+1, num_pcs, model.hidden_dim), dtype=torch.double)
    proj_samples = torch.zeros((duration+1, num_samples, num_pcs), dtype=torch.double)

    pwl_prod.fill(sample_points, sample_labels)

    def on_hidacts(acts_info: RNNHiddenActivations):
        hidden_acts = acts_info.hidden_acts
        recur_step = acts_info.recur_step

        hid_acts[recur_step, :, :] = hidden_acts.detach()
        pc_vals, pc_vecs = get_hidden_pcs(hid_acts[recur_step], num_pcs)
        hid_pc_vals[recur_step, :] = pc_vals
        hid_pc_vecs[recur_step, :, :] = pc_vecs

    model(sample_points, duration, on_hidacts, 1)

    for recur_step in range(duration+1):
        project_to_pcs(hid_acts[recur_step], hid_pc_vecs[recur_step], out=proj_samples[recur_step])

    # We are free to rotate the pc vectors as we please. The following rotates
    # them such that the mean value of each label on each pc stays on the same
    # side. This is only 100% accomplishable for 2 labels since we must swap
    # the direction for ALL labels for a particular pc. we ensure that at least
    # 50% of the labels did not change direction from the start
    indices_by_label = dict(
        (lbl, sample_labels == lbl) for lbl in range(pwl_prod.output_dim)
    )

    means_by_label_and_recur = dict()
    for lbl in range(pwl_prod.output_dim):
        for recur_step in range(duration+1):
            means_by_label_and_recur[(lbl, recur_step)] = (
                proj_samples[recur_step, indices_by_label[lbl], :].mean(0)
            )

    # we can flip any of our pcs
    for recur_step in range(1, duration+1):
        for pc in range(num_pcs): # pylint: disable=invalid-name
            badness = 0
            counter = 0
            for lbl1 in range(pwl_prod.output_dim):
                for lbl2 in range(lbl1 + 1, pwl_prod.output_dim):
                    used_to_be_lt = (
                        means_by_label_and_recur[(lbl1, 0)][pc]
                        < means_by_label_and_recur[(lbl2, 0)][pc]
                    )
                    curr_is_lt = (
                        means_by_label_and_recur[(lbl1, recur_step)][pc]
                        < means_by_label_and_recur[(lbl2, recur_step)][pc]
                    )
                    counter += 1
                    if used_to_be_lt != curr_is_lt:
                        badness += 1

            if badness >= (counter / 2):
                hid_pc_vecs[recur_step, pc, :] *= -1


    for recur_step in range(duration+1):
        project_to_pcs(hid_acts[recur_step], hid_pc_vecs[recur_step], out=proj_samples[recur_step])

    return PCTrajectory(hid_pc_vecs, hid_pc_vals, proj_samples, sample_labels, duration)

def plot_snapshot(axis: plt.Axes, projected: torch.tensor, labels: torch.tensor,
                  min_x: float, max_x: float, min_y: float, max_y: float,
                  alpha: float = 0.5, s: int = 1, cmap='Set1') -> None:
    """Plots the given projected points to the given matplotlib axis.

    Args:
        axis (plt.Axes): The axes to plot the points on
        projected (torch.tensor): a tensor that is [num_samples, 2]
        labels (torch.tensor): a tensor that is [num_samples]

        min_x (float): the left edge of the graph (x: pc 0)
        max_x (float): the right edge of the graph (x: pc 0)
        min_y (float): the top edge of the graph (y: pc 1)
        max_y (float): the bottom edge of the graph (y: pc 1)

        alpha (float): the alpha for the points
    """

    if not torch.is_tensor(projected):
        raise ValueError(f'expected projected is torch.tensor, got {projected}')
    if len(projected.shape) != 2 or projected.shape[1] != 2:
        raise ValueError(f'expected projected.shape is (num_samples, 2), got ({projected.shape})')
    if not torch.is_tensor(labels):
        raise ValueError(f'expected labels is torch.tensor, got {labels}')
    if len(labels.shape) != 1:
        raise ValueError(f'expected labels.shape is (num_samples), got {labels.shape}')
    if projected.shape[0] != labels.shape[0]:
        raise ValueError(f'expected projected.shape is (num_samples, 2) and labels.shape is (num_samples), but projected.shape={projected.shape} and labels.shape={labels.shape} (dont match on dim 0)')
    if not isinstance(min_x, float):
        raise ValueError(f'expected min_x is float, got {min_x} (type={type(min_x)})')
    if not isinstance(max_x, float):
        raise ValueError(f'expected max_x is float, got {max_x} (type={type(max_x)})')
    if min_x > max_x:
        raise ValueError(f'expected min_x < max_x, but min_x={min_x} and max_x={max_x}')
    if not isinstance(min_y, float):
        raise ValueError(f'expected min_y is float, got {min_y} (type={type(min_y)})')
    if not isinstance(max_y, float):
        raise ValueError(f'expected max_y is float, got {max_y} (type={type(max_y)})')
    if min_y > max_y:
        raise ValueError(f'expected min_y < max_y, but min_y={min_y} and max_y={max_y}')
    if not isinstance(alpha, float):
        raise ValueError(f'expected alpha is float, got {alpha}')

    axis.scatter(projected[:, 0].numpy(), projected[:, 1].numpy(),
                 s=s, alpha=alpha, c=labels.numpy(),
                 cmap=mpl.cm.get_cmap(cmap))

    axis.set_xlim([min_x, max_x])
    axis.set_ylim([min_y, max_y])
    axis.tick_params(
        axis='both',
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)

def plot_trajectory(traj: PCTrajectory, filepath: str, exist_ok: bool = False):
    """Plots the given trajectory and saves it to the given filepath.

    Args:
        traj (PCTrajectory): the trajectory to plot
        filepath (str): where to save the images and data. should have extension 'zip' or no extension
        exist_ok (bool, default false): if true we will overwrite existing files rather than error
    """

    if not isinstance(traj, PCTrajectory):
        raise ValueError(f'expected traj is PCTrajectory, got {traj} (type={type(traj)})')
    if not isinstance(filepath, str):
        raise ValueError(f'expected filepath is str, got {filepath} (type={type(filepath)})')

    filepath_wo_ext = os.path.splitext(filepath)[0]
    if filepath_wo_ext == filepath:
        filepath += '.zip'

    if os.path.exists(filepath_wo_ext):
        raise FileExistsError(f'for filepath {filepath} we require {filepath_wo_ext} is available (already exists)')

    if not exist_ok and os.path.exists(filepath):
        raise FileExistsError(f'filepath {filepath} already exists (use exist_ok=True to overwrite)')

    os.makedirs(filepath_wo_ext)

    closest_square: int = int(np.ceil(np.sqrt(traj.get_num_timesteps())))
    num_cols: int = int(math.ceil(traj.get_num_timesteps() / closest_square))
    local_fig, local_axs = plt.subplots(num_cols, closest_square, squeeze=False)
    global_fig, global_axs = plt.subplots(num_cols, closest_square, squeeze=False)

    global_min_x = torch.min(traj.projected_samples[:, :, 0]).item()
    global_min_y = torch.min(traj.projected_samples[:, :, 1]).item()
    global_max_x = torch.max(traj.projected_samples[:, :, 0]).item()
    global_max_y = torch.max(traj.projected_samples[:, :, 1]).item()
    global_padding_x = (global_max_x - global_min_x) * .1
    global_padding_y = (global_max_y - global_min_y) * .1

    timestep: int = 0
    for x in range(num_cols):
        for y in range(closest_square):
            if timestep >= traj.get_num_timesteps():
                local_axs[x][y].remove()
                global_axs[x][y].remove()
                continue
            projected: torch.tensor = traj.projected_samples[timestep, :, :2]
            projected_lbls: torch.tensor = traj.projected_sample_labels

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
                          min_y - padding_y, max_y + padding_y)

            plot_snapshot(global_axs[x][y], projected, projected_lbls,
                          global_min_x - global_padding_x,
                          global_max_x + global_padding_x,
                          global_min_y - global_padding_y,
                          global_max_y + global_padding_y)

            timestep += 1

    local_path = os.path.join(filepath_wo_ext, 'local.png')
    global_path = os.path.join(filepath_wo_ext, 'global.png')
    data_path = os.path.join(filepath_wo_ext, 'data.npz')

    local_fig.tight_layout()
    global_fig.tight_layout()

    local_fig.savefig(local_path, transparent=True)
    global_fig.savefig(global_path, transparent=True)

    np.savez(data_path, principal_vectors=traj.principal_vectors.numpy(),
             principal_values=traj.principal_values.numpy(),
             projected_samples=traj.projected_samples.numpy(),
             projected_sample_labels=traj.projected_sample_labels.numpy())

    counter = 0
    local_exists, global_exists, data_exists = os.path.exists(local_path), os.path.exists(global_path), os.path.exists(data_path)
    while not local_exists or not global_exists or not data_exists:
        if counter == 1:
            warnings.warn(f'waiting on either {local_path} (exists: {local_exists}) or {global_path} (exists: {global_exists}) or {data_path} (exists: {data_exists})', UserWarning)
        if counter > 10:
            raise Exception(f'savefig failed; local_exists={local_exists}, global_exists={global_exists}, data_exists={data_exists}')
        time.sleep(0.05)
        local_exists, global_exists, data_exists = os.path.exists(local_path), os.path.exists(global_path), os.path.exists(data_path)
        counter += 1

    if os.path.exists(filepath):
        os.remove(filepath)

    cwd = os.getcwd()
    shutil.make_archive(filepath_wo_ext, 'zip', filepath_wo_ext)
    os.chdir(cwd)
    shutil.rmtree(filepath_wo_ext)
    os.chdir(cwd)
