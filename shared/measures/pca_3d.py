"""This module handles plotting 3-D pc plots. This may contain either snapshot or animated
approaches to pc plots. These plots can take significantly longer than the 2D counterparts
and are thus structured to go through the npmp module.
"""

import shared.measures.pca_ff as pca_ff
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from matplotlib.animation import FuncAnimation
import pytweening
from shared.filetools import zipdir
from shared.npmp import NPDigestor

def _plot_npmp(projected_sample_labels: np.ndarray, *args, outfile: str = None, exist_ok=False,
               frame_time: float = 16.67):
    """Structured to be npmp friendly, however not very friendly to use compared
    to the public variants. Simply delegates to _plot_ff_real

    Arguments:
        projected_sample_labels (np.ndarray): the sample labels that were projected
            through the layers

        args: Should have a length which is a multiple of 3 where each set is
            (vectors, values, projected_samples)

        outfile (str): expected to come from the keyword arguments. not optional
        exist_ok (bool): if we should overwrite the file if it already exists, default False
    """
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok} (type={type(exist_ok)})')

    proj_sample_torch = torch.from_numpy(projected_sample_labels)

    snapshots = []
    for i in range(len(args) // 3):
        princ_vectors = args[i*3]
        princ_values = args[i*3 + 1]
        proj_samples = args[i*3 + 2]

        snapshots.append(pca_ff.PCTrajectoryFFSnapshot(
            torch.from_numpy(princ_vectors),
            torch.from_numpy(princ_values),
            torch.from_numpy(proj_samples),
            proj_sample_torch
        ))

    traj = pca_ff.PCTrajectoryFF(snapshots)
    _plot_ff_real(traj, outfile, exist_ok, frame_time=frame_time)

def _plot_ff_real(traj: pca_ff.PCTrajectoryFF, outfile: str, exist_ok: bool,
                  frame_time: float = 16.67):
    """Plots the given feed-forward pc trajectory

    Args:
        traj (pca_ff.PCTrajectoryFF): the trajectory to plot
        outfile (str): a path to the zip file we should save the plots in
        exist_ok (bool): true to overwrite existing zip, false to keep it
    """
    if not isinstance(traj, pca_ff.PCTrajectoryFF):
        raise ValueError(f'expected traj is pca_ff.PCTrajectoryFF, got {traj} (type={type(traj)})')
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok} (type={type(exist_ok)})')

    outfile_wo_ext = os.path.splitext(outfile)[0]
    if outfile == outfile_wo_ext:
        outfile += '.zip'

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'in order to save {outfile} need {outfile_wo_ext} as working space')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'cannot save {outfile} (already exists) (set exist_ok=True to overwrite)')

    os.makedirs(outfile_wo_ext, exist_ok=exist_ok)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _visible_layer = None
    _scatter = None
    def rotate_xz(perc, force=None):
        angle = 45 + 360 * perc if force is None else 45 + force
        ax.view_init(30, angle)
        return ax

    def movetime(perc, force=None, norotate=False):
        nonlocal _visible_layer, _scatter

        target_layer = force if force is not None else int(perc * traj.num_layers)
        if target_layer == traj.num_layers:
            target_layer -= 1

        if target_layer == _visible_layer:
            return tuple()

        _visible_layer = target_layer
        snapsh: pca_ff.PCTrajectoryFFSnapshot = traj.snapshots[target_layer]
        if _scatter is None:
            _scatter = ax.scatter(snapsh.projected_samples[:, 0].numpy(),
                                  snapsh.projected_samples[:, 1].numpy(),
                                  snapsh.projected_samples[:, 2].numpy(),
                                  s=1,
                                  c=snapsh.projected_sample_labels.numpy())
            if not norotate:
                ax.view_init(30, 45)
            return (ax, _scatter)

        _scatter._offsets3d = (snapsh.projected_samples[:, 0].numpy(),
                               snapsh.projected_samples[:, 1].numpy(),
                               snapsh.projected_samples[:, 2].numpy())
        return (ax, _scatter)

    def _updater(time_ms: float, start_ms: int, end_ms: int, easing, target, on_first=None):
        if time_ms < start_ms:
            return False, None
        if time_ms > end_ms:
            return False, None
        progress = (time_ms - start_ms) / (end_ms - start_ms)
        first = time_ms - frame_time <= start_ms
        if first and on_first:
            on_first()

        return True, target(easing(progress))

    actions = [
        (5000, (pytweening.linear, movetime)),
        (5000, (lambda x: 1-x, movetime)),
        (5000, (pytweening.easeInOutSine, rotate_xz))
    ]

    def reglyr(lyr):
        actions.append((5000, (pytweening.easeInOutSine, rotate_xz, lambda: movetime(0, lyr))))

    for lyr in range(1, traj.num_layers):
        reglyr(lyr)

    total_time = sum(act[0] for act in actions)
    def update(time_ms: float):
        start = 0
        for act in actions:
            succ, res = _updater(time_ms, start, act[0] + start, *act[1])
            if succ:
                return res
            start += act[0]
        return tuple()

    movetime(0)

    anim = FuncAnimation(fig, update, frames=np.arange(0, total_time+1, frame_time), interval=frame_time)
    #anim.save(os.path.join(outfile_wo_ext, 'out.mp4'), dpi=100, writer='ffmpeg')

    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _visible_layer = None # force redraw
    _scatter = None

    snapshotdir = os.path.join(outfile_wo_ext, 'snapshots')
    os.makedirs(snapshotdir)

    for lyr in range(traj.num_layers):
        for angle in (15, 0, -15, 90, 180, 270):
            movetime(0, force=lyr, norotate=True)
            rotate_xz(0, angle)
            fig.savefig(os.path.join(snapshotdir, f'snapshot_{angle+45}_{lyr}.png'))

    plt.close(fig)

    if os.path.exists(outfile):
        os.remove(outfile)
    zipdir(outfile_wo_ext)

def plot_ff(traj: pca_ff.PCTrajectoryFF, outfile: str, exist_ok: bool,
            frame_time: float = 16.67, digestor: NPDigestor = None):
    """Plots the given trajectory to the given outfile if possible. If the
    digestor is given, then this effect takes place on a different thread

    Arguments:
        traj (pca_ff.PCTrajectoryFF): the trajectory to plot
        outfile (str): where to plot the trajectory
        exist_ok (bool): True to overwrite existing plot, false to error if it exists
        digestor (NPDigestor, optional): Default None. If specified, this will be used
            to multiprocess the operation
    """
    if not isinstance(traj, pca_ff.PCTrajectoryFF):
        raise ValueError(f'expected traj is PCTrajectoryFF, got {traj} (type={type(traj)})')
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok} (type={type(exist_ok)})')
    if digestor is not None and not callable(digestor):
        raise ValueError(f'expected digestor is callable, got {digestor} (type={type(digestor)})')

    outfile_wo_ext = os.path.splitext(outfile)[0]
    if outfile == outfile_wo_ext:
        outfile += '.zip'

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'to write {outfile} need {outfile_wo_ext} available')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'outfile {outfile} already exists (use exist_ok=True to overwrite)')

    if digestor is None:
        _plot_ff_real(traj, outfile, exist_ok)
        return

    sample_labels = traj.snapshots[0].projected_sample_labels.numpy()
    args = []
    for snapshot in traj.snapshots:
        snapshot: pca_ff.PCTrajectoryFFSnapshot
        args.append(snapshot.principal_vectors.numpy())
        args.append(snapshot.principal_values.numpy())
        args.append(snapshot.projected_samples.numpy())
    digestor(sample_labels, *args, outfile=outfile, exist_ok=exist_ok,
             frame_time=frame_time, target_module='shared.measures.pca_3d',
             target_name='_plot_npmp')
