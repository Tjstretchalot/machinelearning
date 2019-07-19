"""This module handles plotting 3-D pc plots. This may contain either snapshot or animated
approaches to pc plots. These plots can take significantly longer than the 2D counterparts
and are thus structured to go through the npmp module.
"""

import shared.measures.pca_ff as pca_ff
import shared.measures.pca_gen as pca_gen
import numpy as np
import torch
import os
import traceback
import io
import sys
import typing
import bisect
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import pytweening
import shared.mytweening as mytweening
from shared.filetools import zipdir
from shared.npmp import NPDigestor
from shared.trainer import GenericTrainingContext
import shared.myqueue as myq
import shared.async_anim as saa
import shared.typeutils as tus
import shared.measures.utils as mutils
import shared.measures.clusters as clusters
import time as time_
from multiprocessing import Process
import queue

FRAME_SIZE = (19.2, 10.8) # oh baby
DPI = 100 # 100 -> 2k, 200 -> 4k

INPUT_SPIN_TIME = 12500
OTHER_SPIN_TIME = 10000
INTERP_SPIN_TIME = 6000
ZOOM_TIME = 2000
CLUSTER_ZOOM_TIME = 2000
CLUSTER_SPIN_TIME = 10000

NUM_WORKERS = 6#3
FRAMES_PER_SYNC = 10

PRINT_EVERY = 15

def _plot_npmp(projected_sample_labels: np.ndarray, *args, outfile: str = None, exist_ok=False,
               **kwargs):
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
    _plot_ff_real(traj, outfile, exist_ok, **kwargs)

class MPLData:
    """MPL references for the videos

    Attributes:
        figure (mpl.figure.Figure): the figure
        axes (mpl.axes.Axes): the 3D axes
        title (mpl.text.Text): the text object for the title
        scatter (?): the 3d scatter reference

        current_snapsh_idx (int): the snapshot index that is currently visible
    """

    def __init__(self, figure, axes, title, scatter, current_snapsh_idx):
        self.figure = figure
        self.axes = axes
        self.title = title
        self.scatter = scatter
        self.current_snapsh_idx = current_snapsh_idx

class Scene:
    """Describes a scene which takes some time to perform

    Attributes:
        duration (int); the number of milliseconds the scene lasts
        title (str): the title for this scene
    """

    def __init__(self, duration: int, title: str):
        self.duration = duration
        self.title = title

    def start(self, traj, mpl_data):
        """Called before any frames start rendering"""
        pass

    def apply(self, traj, mpl_data, time_ms):
        """Applies this scene to the matplotlib figure
        """
        if mpl_data.title.get_text() != self.title:
            mpl_data.title.set_text(self.title)

    def finish(self, traj, mpl_data):
        """Called after all frames have finished rendering"""
        pass

class ZoomScene(Scene):
    """Scene which changes the zoom

    Attributes:
        start_zoom (tuple[float, float]): the min/max we start at
        end_zoom (tuple[float, float]): the min/max we end at
        snapshot_idx (int): the snapshot that should be visible
    """

    def __init__(self, duration, title, start_zoom, end_zoom, snapshot_idx):
        super().__init__(duration, title)
        self.start_zoom = start_zoom
        self.end_zoom = end_zoom
        self.snapshot_idx = snapshot_idx

    def apply(self, traj, mpl_data, time_ms):
        if mpl_data.current_snapsh_idx != self.snapshot_idx:
            data = traj.snapshots[self.snapshot_idx].projected_samples[:, :3].numpy()
            mpl_data.scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2]) # pylint: disable=protected-access
            mpl_data.current_snapsh_idx = self.snapshot_idx

        prog = time_ms / self.duration
        perc = mytweening.smoothstep(mytweening.squeeze(prog, 0.1))

        delzoom = (self.end_zoom[0] - self.start_zoom[0], self.end_zoom[1] - self.start_zoom[1])
        newzoom = (self.start_zoom[0] + delzoom[0] * perc, self.start_zoom[1] + delzoom[1] * perc)

        mpl_data.axes.set_xlim(*newzoom)
        mpl_data.axes.set_ylim(*newzoom)
        mpl_data.axes.set_zlim(*newzoom)

class RotationScene(Scene):
    """Scene which rotates around the y axis"""
    def __init__(self, duration, title, snapshot_idx):
        super().__init__(duration, title)
        self.snapshot_idx = snapshot_idx

    def apply(self, traj, mpl_data, time_ms):
        super().apply(traj, mpl_data, time_ms)

        if mpl_data.current_snapsh_idx != self.snapshot_idx:
            data = traj.snapshots[self.snapshot_idx].projected_samples[:, :3].numpy()
            mpl_data.scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2]) # pylint: disable=protected-access
            minlim, maxlim = float(data.min()), float(data.max())
            mpl_data.axes.set_xlim(minlim, maxlim)
            mpl_data.axes.set_ylim(minlim, maxlim)
            mpl_data.axes.set_zlim(minlim, maxlim)
            mpl_data.current_snapsh_idx = self.snapshot_idx

        prog = time_ms / self.duration
        if prog < 0 or prog > 1:
            raise ValueError(f'time_ms={time_ms}, duration={self.duration}, prog={prog}')
        perc = pytweening.easeInOutSine(prog)
        rot = 45 + 360 * perc
        mpl_data.axes.view_init(30, rot)

class InterpScene(Scene):
    """Scene which interpolates between two snapshots while spinning"""
    def __init__(self, duration, title, from_snapshot_idx, to_snapshot_idx):
        super().__init__(duration, title)
        self.from_snapshot_idx = from_snapshot_idx
        self.to_snapshot_idx = to_snapshot_idx
        self.start_np = None
        self.delta_np = None
        self.lims = None

    def start(self, traj, mpl_data):
        self.start_np = traj.snapshots[self.from_snapshot_idx].projected_samples[:, :3].numpy()
        end_np = traj.snapshots[self.to_snapshot_idx].projected_samples[:, :3].numpy()
        self.delta_np = end_np - self.start_np
        minlim = min(float(self.start_np.min()), float(end_np.min()))
        maxlim = max(float(self.start_np.max()), float(end_np.max()))
        self.lims = (minlim, maxlim)


    def apply(self, traj, mpl_data, time_ms):
        super().apply(traj, mpl_data, time_ms)

        prog = time_ms / self.duration
        if prog < 0 or prog > 1:
            raise ValueError(f'time_ms={time_ms}, duration={self.duration}, prog={prog}')
        rot_perc = pytweening.easeInOutSine(mytweening.squeeze(prog, 0.1))
        interp_perc = pytweening.easeInOutCirc(prog)

        data = self.start_np + self.delta_np * interp_perc
        mpl_data.current_snapsh_idx = -1
        mpl_data.scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2]) # pylint: disable=protected-access
        mpl_data.axes.set_xlim(*self.lims)
        mpl_data.axes.set_ylim(*self.lims)
        mpl_data.axes.set_zlim(*self.lims)

        rot = 45 + 360 * rot_perc
        mpl_data.axes.view_init(30, rot)

    def finish(self, traj, mpl_data):
        self.start_np = None
        self.delta_np = None

class MaskedScatter:
    """Something that acts like a scatter plot from matplotlib
    except that it's masked in such a way that changing the offsets
    corresponds to changing a masked set of offsets.

    Attributes:
        real_scatter (?): the 3d scatter reference
        mask (ndarray[num_samples], bool): the mask on the points
        oth_points (ndarray[num_samples, 3], uint8): the default value for points
            which are not specified.
    """
    def __init__(self, real_scatter, mask, oth_points):
        self.real_scatter = real_scatter
        self.mask = mask
        self.oth_points = oth_points

    def __setattr__(self, name, value):
        if name != '_offsets3d':
            super().__setattr__(name, value)
            return

        maskx, masky, maskz = value
        newx = self.oth_points[:, 0].copy()
        newy = self.oth_points[:, 1].copy()
        newz = self.oth_points[:, 2].copy()
        newx[self.mask] = maskx
        newy[self.mask] = masky
        newz[self.mask] = maskz
        self.real_scatter._offsets3d = (newx, newy, newz) # pylint: disable=protected-access


class MaskedMPLData:
    """A wrapper on another MPLData that applies a mask.

    Attributes:
        real_mpl (MPLData): the real mpl data this is masking
        masked_scatter (MaskedScatter): the scatter but masked
    """
    def __init__(self, real_mpl, masked_scatter):
        self.real_mpl = real_mpl
        self.masked_scatter = masked_scatter

    def __getattr__(self, name):
        if name == 'scatter':
            return super().__getattribute__('masked_scatter')
        return getattr(self.real_mpl, name)

class MaskedParentScene(Scene):
    """Describes a scene which accepts child scenes. The children scenes, however,
    see a different trajectory than the overall trajectory. This is initially intended
    to support zooming into clusters.

    Attributes:
        masked_traj (union[pca_ff.PCTrajectoryFF, pca_gen.PCTrajectoryGen]):
            the trajectory that children see
        outer_traj_snapshot_ind (int): the index in the outer snapshot to use
        mask (tensor[num_samples], dtype uint8): the mask on the outer tensor
            that we are using
        masked_mpl (MaskedMPLData): the masked mpl data, initialized on start
        children (list[Scene]): the children scenes
        children_end_times (list[int]): when the children scenes end
    """
    def __init__(self, masked_traj, outer_traj_snapshot_ind, mask, children):
        super().__init__(sum((child.duration for child in children), 0), 'MaskedParentScene')
        self.masked_traj = masked_traj
        self.outer_traj_snapshot_ind = outer_traj_snapshot_ind
        self.mask = mask
        self.masked_mpl = None
        self.children = children

        self.children_end_times = []
        cursum = 0
        for child in self.children:
            cursum += child.duration
            self.children_end_times.append(cursum)

    def start(self, traj, mpl_data: MPLData):
        self.masked_mpl = MaskedMPLData(
            mpl_data,
            MaskedScatter(
                mpl_data.scatter,
                self.mask,
                traj.snapshots[self.outer_traj_snapshot_ind].projected_samples
                ))
        for child in self.children:
            child.start(self.masked_traj, self.masked_mpl)

    def _get_scene_and_time(self, time_ms):
        millis = time_ms

        scene_loc = bisect.bisect_right(self.children_end_times, millis)
        if scene_loc < 0 or scene_loc >= len(self.children):
            raise ValueError(f'there is no scene at time {time_ms}')
        prev_end_time = self.children_end_times[scene_loc - 1] if scene_loc > 0 else 0
        return self.children[scene_loc], millis - prev_end_time

    def apply(self, traj, mpl_data, time_ms):
        child, millis = self._get_scene_and_time(time_ms)
        child.apply(self.masked_traj, self.masked_mpl, millis)

    def finish(self, traj, mpl_data):
        for child in self.children:
            child.finish(self.masked_traj, self.masked_mpl)

class FrameWorker:
    """Describes a frame worker. They are initialized each with enough information to make
    any frame of the video. They are then told which frames to send by the main thread.
    Using frame workers can ensure we are bottlenecked only by ffmpeg's ability to save
    to file

    Attributes:
        img_queue (Queue): the queue we push images to
        rec_queue (Queue): the queue we receive messages from
            image message: ('img', <frame number>)
            sync message: ('sync', time.time())
            finish message: ('finish',)
        send_queue (Queue): the queue we can inform the main thread with
            sync message: ('sync', time.time())

        ms_per_frame (float): the number of milliseconds per frame
        frame_size (tuple[float, float]): size of the frame in inches
        dpi (int): number of pixels per inch
        scenes (list[Scene]): the scenes which this worker has
        scene_end_times (list[int]): a strictly increasing list of the times at which
            each scene ends, for performance.

        traj (pca_ff.PCTrajectoryFF): the trajectory we are plotting
        s (float): the size of the markers

        mpl_data (MPLData): the actual mpl data we have
    """

    def __init__(self, img_queue, rec_queue, send_queue, ms_per_frame,
                 frame_size, dpi, scenes, traj, s):
        self.img_queue = img_queue
        self.rec_queue = rec_queue
        self.send_queue = send_queue
        self.ms_per_frame = ms_per_frame
        self.frame_size = frame_size
        self.dpi = dpi
        self.scenes = scenes
        self.traj = traj
        self.s = s

        self.mpl_data = None

        self.scene_end_times = []
        sumtime = 0
        for scene in scenes:
            sumtime += scene.duration
            self.scene_end_times.append(sumtime)

    def init_mpl(self):
        """Initializes the figure, axes, title, and scatter plot"""
        fig = plt.figure(figsize=self.frame_size)
        ax = fig.add_subplot(111, projection='3d')
        axtitle = ax.set_title('Title')
        axtitle.set_fontsize(80)

        data = self.traj.snapshots[0].projected_samples[:, :3].numpy()
        labels = self.traj.snapshots[0].projected_sample_labels.numpy()
        scatter = self.init_scatter(ax, data, labels)
        ax.view_init(30, 45)

        minlim = float(data.min())
        maxlim = float(data.max())
        ax.set_xlim(minlim, maxlim)
        ax.set_ylim(minlim, maxlim)
        ax.set_zlim(minlim, maxlim)

        self.mpl_data = MPLData(fig, ax, axtitle, scatter, 0)

    def init_scatter(self, ax, data, labels):
        """Initializes the scatter plot for the given data and labels"""
        return ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                          s=self.s, c=labels, cmap=mpl.cm.get_cmap('Set1'))

    def start_scenes(self):
        """Starts all the scenes"""
        for scene in self.scenes:
            scene.start(self.traj, self.mpl_data)

    def finish_scenes(self):
        """Finishes all the scenes"""
        for scene in self.scenes:
            scene.finish(self.traj, self.mpl_data)

    def get_scene_and_time(self, frame_num):
        """Determines the which scene and when in the scene the given frame is"""
        millis = frame_num * self.ms_per_frame

        scene_loc = bisect.bisect_right(self.scene_end_times, millis)
        if scene_loc < 0 or scene_loc >= len(self.scenes):
            raise ValueError(f'there is no frame {frame_num}')
        prev_end_time = self.scene_end_times[scene_loc - 1] if scene_loc > 0 else 0
        return self.scenes[scene_loc], millis - prev_end_time

    def do_all(self):
        """This is meant to be the function that is invoked immediately after initialization,
        which finishes preparing and then reads from the main thread performing actions until
        completion"""

        self.init_mpl()
        self.start_scenes()

        while True:
            msg = self.rec_queue.get()
            if msg[0] == 'sync':
                self.send_queue.put(('sync', time_.time()))
                continue
            if msg[0] == 'finish':
                break
            if msg[0] != 'img':
                raise ValueError(f'unexpected message: {msg}')

            frame_num = msg[1]
            scene, time = self.get_scene_and_time(frame_num)
            scene.apply(self.traj, self.mpl_data, time)
            hndl = io.BytesIO()
            self.mpl_data.figure.set_size_inches(*self.frame_size)
            self.mpl_data.figure.savefig(hndl, format='rgba', dpi=self.dpi)

            rawimg = hndl.getvalue()
            self.img_queue.put((frame_num, rawimg))
            del hndl

        self.finish_scenes()

        self.img_queue.close()
        self.send_queue.close()
        self.rec_queue.close()

class ConcattedScatter:
    """A wrapper around multiple scatter plots to make them appear like
    a single scatter plot

    Attributes:
        scatters (list[mask, scatter]): the scatter plots we are wrapping
    """
    def __init__(self, scatters):
        self.scatters = scatters

    def __setattr__(self, name, value):
        if name == 'scatters':
            super().__setattr__(name, value)
            return

        if name == '_offsets3d':
            xs, ys, zs = value
            for mask, scatter in self.scatters:
                scatter._offsets3d = (xs[mask], ys[mask], zs[mask]) # pylint: disable=protected-access
            return

        raise AttributeError(f'no such attribute: {name}')


class GenFrameWorker(FrameWorker):
    """A frame worker which accepts a general pc trajectory and the associated
    required information for determining the color and marker for each point.

    Attributes (beyond or changed from FrameWorker):
        traj (pca_gen.PCTrajectoryGen): the trajectory to plot
        markers (callable): accepts an ndarray of shape (batch_size, output_size)
            and returns a iterable of (ndarray[batch_size] dtype=bool, str) where the
            ndarray is a mask for the points and the str is the marker to use
        scalar_mapping (OutputToScalarMapping): maps the outputs (batch_size, output_size)
            to scalars which can be sent through the norm and then the color map
        norm (mcolors.Normalize): rescale the result from the OutputToScalarMapping
            to 0-1
        cmap (union[ColorMap, str]): either a colormap or a string identifier for
            a colormap that maps the 0-1 output of the norm to colors
    """
    def __init__(self, img_queue, rec_queue, send_queue, ms_per_frame,
                 frame_size, dpi, scenes, traj, s, markers, scalar_mapping,
                 norm, cmap):
        super().__init__(img_queue, rec_queue, send_queue, ms_per_frame,
                         frame_size, dpi, scenes, traj, s)
        self.markers = markers
        self.scalar_mapping = scalar_mapping
        self.norm = norm
        self.cmap = cmap

    def init_scatter(self, ax, data, labels):
        return _init_scatter_gen(self.scalar_mapping, self.cmap, self.norm, self.markers,
                                 ax, data, labels, self.s)

def _init_scatter_gen(scalar_mapping, cmap, norm, markers, ax, data, labels, s):
    scatters = []

    for mask, marker in markers(labels):
        masked_data = data[mask]
        marker = marker if isinstance(marker, dict) else {'marker': marker}
        scatters.append(
            (mask,
             ax.scatter(masked_data[:, 0], masked_data[:, 1], masked_data[:, 2],
                        s=s, c=scalar_mapping(torch.from_numpy(labels[mask])),
                        cmap=mcm.get_cmap(cmap),
                        norm=norm, **marker))
        )

    return ConcattedScatter(scatters)

def _frame_worker_target(img_queue, rec_queue, send_queue, ms_per_frame, frame_size, dpi,
                         scenes, traj, s, logfile):
    worker = FrameWorker(
        myq.ZeroMQQueue.deser(img_queue), myq.ZeroMQQueue.deser(rec_queue),
        myq.ZeroMQQueue.deser(send_queue), ms_per_frame, frame_size, dpi, scenes, traj, s)

    try:
        worker.do_all()
    except:
        traceback.print_exc()
        with open(logfile, 'r') as infile:
            traceback.print_exc(file=infile)
        raise

def _frame_worker_target_gen(img_queue, rec_queue, send_queue, ms_per_frame, frame_size,
                             dpi, scenes, traj, s, markers, scalar_mapping, norm, cmap, logfile):
    """Target for frame workers using the GenFrameWorker subclass

    Args:
        img_queue (ZeroMQQueue, serialized): the queue to send images through
        rec_queue (ZeroMQQueue, serialized): the queue to receive jobs from
        send_queue (ZeroMQQueue, serialized): the queue to notify jobs are complete
        ms_per_frame (float): milliseconds per frame
        frame_size (tuple[float, float]): size of the frame in inches
        dpi (int): pixels per inch
        scenes (list[Scene]): the scenes we are trying to render.
        traj (PCTrajectoryGen): the trajectory we are rending
        s (float): the size of the markers
        markers (str): the path to the module and name of the marker producing callable
        scalar_mapping (str): the path to the module and name of the scalar mapping
            producing callable
        norm (str): the path to the module and name of the norm producing callable
        cmap (str): the name of the color map to use
        logfile (str): where to store errors if they occur
    """
    worker = GenFrameWorker(
        myq.ZeroMQQueue.deser(img_queue), myq.ZeroMQQueue.deser(rec_queue),
        myq.ZeroMQQueue.deser(send_queue), ms_per_frame, frame_size, dpi, scenes, traj, s,
        mutils.get_fixed_single(markers)(), mutils.get_fixed_single(scalar_mapping)(),
        mutils.get_fixed_single(norm)(), cmap
    )

    try:
        worker.do_all()
    except:
        traceback.print_exc()
        with open(logfile, 'r') as infile:
            traceback.print_exc(file=infile)
        raise

class FrameWorkerConnection:
    """An instance in the main thread that describes a connection with a frame worker

    Attributes:
        proc (Process): the actual child process
        img_queue (queue): the queue the frame worker sends images to us with
        send_queue (queue): the queue we send the frame worker messages with
        ack_queue (queue): the queue we receive messages from the frame worker from
        awaiting_sync (bool): True if we are awaiting a sync message, false otherwise
    """

    def __init__(self, proc: Process, img_queue, send_queue, ack_queue):
        self.proc = proc
        self.img_queue = img_queue
        self.send_queue = send_queue
        self.ack_queue = ack_queue
        self.awaiting_sync = False

    def start_sync(self):
        """Starts the syncing process"""
        self.send_queue.put(('sync', time_.time()))
        self.awaiting_sync = True

    def check_sync(self):
        """Checks if the syncing process is complete"""
        if not self.awaiting_sync:
            return True
        try:
            ack = self.ack_queue.get_nowait()
            if ack[0] != 'sync':
                raise ValueError(f'expected sync response, got {ack}')
            sync_time = time_.time() - ack[1]
            if sync_time > 1:
                print(f'[FrameWorkerConnection] took a long time to sync ({sync_time:.3f} s)')
            self.awaiting_sync = False
            return True
        except queue.Empty:
            return False

    def sync(self):
        """Waits for this worker to catch up"""
        self.start_sync()
        resp = self.ack_queue.get()
        self.awaiting_sync = False
        if resp[0] != 'sync':
            raise ValueError(f'expected sync response, got {resp}')
        sync_time = time_.time() - resp[1]
        if sync_time > 1:
            print(f'[FrameWorkerConnection] took a long time to sync ({sync_time:.3f} s)')
        return sync_time

    def start_finish(self):
        """Starts the finish process"""
        self.send_queue.put(('finish',))

    def check_finish(self):
        """Checks if the worker has shutdown yet"""
        return self.proc.is_alive()

    def wait_finish(self):
        """Waits for finish process to complete"""
        self.proc.join()

    def finish(self):
        """Cleanly shutdowns the worker"""
        self.start_finish()
        self.wait_finish()

    def send(self, frame_num):
        """Notifies this worker that it should render the specified frame number"""
        self.send_queue.put(('img', frame_num))

def _get_cluster_scene(traj: typing.Union[pca_ff.PCTrajectoryFF, pca_gen.PCTrajectoryGen],
                       layer_ind: int, layer_name: str, clust: clusters.Clusters,
                       clust_ind: int) -> Scene:
    rel_snap = traj.snapshots[layer_ind]
    mask = torch.from_numpy(clust.labels) == clust_ind
    if isinstance(rel_snap, pca_ff.PCTrajectoryFFSnapshot):
        rel_snap: pca_ff.PCTrajectoryFFSnapshot

        new_snap = pca_ff.PCTrajectoryFFSnapshot(
            rel_snap.principal_vectors, rel_snap.principal_values,
            rel_snap.projected_samples[mask].contiguous(),
            rel_snap.projected_sample_labels[mask].contiguous()
        )
        new_traj = pca_ff.PCTrajectoryFF([new_snap])
    else:
        rel_snap: pca_gen.PCTrajectoryGenSnapshot

        new_snap = pca_gen.PCTrajectoryGenSnapshot(
            rel_snap.principal_vectors, rel_snap.principal_values,
            rel_snap.projected_samples[mask].contiguous(),
            rel_snap.projected_sample_labels[mask].contiguous()
        )
        new_traj = pca_gen.PCTrajectoryGen([new_snap])

    old_zoom = (
        rel_snap.projected_samples.min().item(),
        rel_snap.projected_samples.max().item()
    )
    new_zoom = (
        new_snap.projected_samples.min().item(),
        new_snap.projected_samples.max().item()
    )

    title = f'{layer_name} Cluster {clust_ind+1}'
    return MaskedParentScene(new_traj, layer_ind, mask.numpy().astype('bool'), [
        ZoomScene(CLUSTER_ZOOM_TIME, title, old_zoom, new_zoom, 0),
        RotationScene(CLUSTER_SPIN_TIME, title, 0),
        ZoomScene(CLUSTER_ZOOM_TIME, title, new_zoom, old_zoom, 0)
    ])


def _plot_ff_real(traj: typing.Union[pca_ff.PCTrajectoryFF, pca_gen.PCTrajectoryGen],
                  outfile: str, exist_ok: bool,
                  frame_time: float = 16.67, layer_names: typing.Optional[typing.List] = None,
                  snapshots: typing.Optional[typing.List[typing.Tuple[int, int, dict]]] = None,
                  video: bool = True,
                  markers: str = None, scalar_mapping: str = None, norm: str = None,
                  cmap: str = None, s: float = 3,
                  clusts: typing.Optional[
                      typing.List[typing.Optional[clusters.Clusters]]] = None):
    """Plots the given feed-forward pc trajectory

    Args:
        traj (union[pca_ff.PCTrajectoryFF, pca_gen.PCTrajectoryGen]): the trajectory to plot
        outfile (str): a path to the zip file we should save the plots in
        exist_ok (bool): true to overwrite existing zip, false to keep it
        snapshots (list[tuple[int, int, dict]]): a list of snapshots to take, where a snapshot
            is described with (layer, angle, savefig_kwargs). Defaults to many many screenshots
            with a reasonable balance between performance and aesthetics. May be set to an empty
            list for no snapshots
        video (bool): if True a video is rendered, if False, just snapshots are rendered

        markers (str): if the trajectory is a PCTrajectoryGen, this is the path to the module and
            callable that returns a callable which accepts a tensor from the output layer and
            returns an iterable of [tensor, str] where the tensor is a mask for the samples
            and the str is the marker that should be used.
            Alternatively, instead of a str, the marker may be a dict which is simply passed
            as keyword-arguments to the plot (replicating marker with {'marker': marker})
        scalar_mapping (str): if the trajectory is a PCTrajectoryGen, this is the path to the
            module and callable that returns a OutputToScalarMapping-style callable.
        norm (str): if the trajectory is a PCTrajectoryGen, this is the path to the module and
            callable that returns a matplotlib.colors.Normalize instance or subclass instance
        cmap (str): The colormap to use. Defaults to 'Set1' for a PCTrajectoryFF and 'cividis'
            for PCTrajectoryGen
            NOTE: currently cmap on PCTrajectoryFF only effects the snapshots
        s (float): the size of the markers
        clusts (list[optional[Clusters]], optional): if provided, it must have the same length as
            there are layers in the trajectory. For each corresponding index with clusters, a zoom
            in to the cluster will be shown in the final video.

    """
    if not isinstance(traj, (pca_ff.PCTrajectoryFF, pca_gen.PCTrajectoryGen)):
        raise ValueError(f'expected traj is pca_ff.PCTrajectoryFF, got {traj} (type={type(traj)})')
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok} (type={type(exist_ok)})')
    if layer_names is not None:
        if not isinstance(layer_names, (tuple, list)):
            raise ValueError(f'expected layer_names is tuple or list, got {layer_names} (type={type(layer_names)})')
        if len(layer_names) != traj.num_layers:
            raise ValueError(f'expected len(layer_names) = traj.num_layers = {traj.num_layers}, got {len(layer_names)}')

    if isinstance(traj, pca_gen.PCTrajectoryGen):
        tus.check(markers=(markers, str), scalar_mapping=(scalar_mapping, str),
                  norm=(norm, str))
        cmap = cmap or 'cividis'
    else:
        cmap = cmap or 'Set1'

    if clusts is not None:
        tus.check(clusts=(clusts, (tuple, list)))
        tus.check_list((type(None), clusters.Clusters), clusts=clusts)
        if len(clusts) != traj.num_layers:
            raise ValueError(f'expected clusters is length {traj.num_layers}, got {len(clusts)}')
    else:
        clusts = list(None for i in traj.num_layers)

    if snapshots is None:
        snapshots = []
        for lyr in range(traj.num_layers):
            for angle in (15, 0, -15, 90, 180, 270):
                snapshots.append((lyr, angle, {'dpi': DPI}))
    else:
        for snap in snapshots:
            if 'dpi' not in snap[2]:
                snap[2]['dpi'] = DPI


    fps = int(round(1000 / frame_time))
    frame_time = 1000 / fps

    outfile_wo_ext = os.path.splitext(outfile)[0]
    if outfile == outfile_wo_ext:
        outfile += '.zip'

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'in order to save {outfile} need {outfile_wo_ext} as working space')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'cannot save {outfile} (already exists) (set exist_ok=True to overwrite)')

    os.makedirs(outfile_wo_ext, exist_ok=exist_ok)

    fig = None
    ax = None
    _visible_layer = None
    _scatter = None
    _axtitle = None
    def rotate_xz(perc, force=None):
        angle = 45 + 360 * perc if force is None else 45 + force
        ax.view_init(30, angle)
        return ax

    def movetime(perc, force=None, norotate=False):
        nonlocal _visible_layer, _scatter, _axtitle

        target_layer = force if force is not None else int(perc * traj.num_layers)
        if target_layer == traj.num_layers:
            target_layer -= 1

        if target_layer == _visible_layer:
            return tuple()


        _visible_layer = target_layer
        snapsh: pca_ff.PCTrajectoryFFSnapshot = traj.snapshots[target_layer]
        if _scatter is None:
            if isinstance(traj, pca_ff.PCTrajectoryFF):
                _scatter = ax.scatter(snapsh.projected_samples[:, 0].numpy(),
                                    snapsh.projected_samples[:, 1].numpy(),
                                    snapsh.projected_samples[:, 2].numpy(),
                                    s=s,
                                    c=snapsh.projected_sample_labels.numpy(),
                                    cmap=mpl.cm.get_cmap(cmap))
            else:
                _scatter = _init_scatter_gen(
                    mutils.get_fixed_single(scalar_mapping)(),
                    cmap,
                    mutils.get_fixed_single(norm)(),
                    mutils.get_fixed_single(markers)(),
                    ax,
                    snapsh.projected_samples.numpy(),
                    snapsh.projected_sample_labels.numpy(),
                    s
                )
            if not norotate:
                ax.view_init(30, 45)

            minlim = float(snapsh.projected_samples.min())
            maxlim = float(snapsh.projected_samples.max())
            ax.set_xlim(minlim, maxlim)
            ax.set_ylim(minlim, maxlim)
            ax.set_zlim(minlim, maxlim)

            if layer_names is not None:
                _axtitle = ax.set_title(layer_names[target_layer])
                _axtitle.set_fontsize(80)
            return (ax, _scatter)

        if layer_names is not None:
            _axtitle.set_text(layer_names[target_layer])

        _scatter._offsets3d = (snapsh.projected_samples[:, 0].numpy(),
                               snapsh.projected_samples[:, 1].numpy(),
                               snapsh.projected_samples[:, 2].numpy())

        minlim = float(snapsh.projected_samples.min())
        maxlim = float(snapsh.projected_samples.max())
        ax.set_xlim(minlim, maxlim)
        ax.set_ylim(minlim, maxlim)
        ax.set_zlim(minlim, maxlim)

        if layer_names is not None:
            return (ax, _scatter, _axtitle)
        return (ax, _scatter)

    fig = plt.figure(figsize=FRAME_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    _visible_layer = None # force redraw
    _scatter = None

    snapshotdir = os.path.join(outfile_wo_ext, 'snapshots')
    os.makedirs(snapshotdir)

    for lyr, angle, sf_kwargs in snapshots:
        movetime(0, force=lyr, norotate=True)
        rotate_xz(0, angle)
        filetype = 'png' if ('format' not in sf_kwargs) else sf_kwargs['format']
        fig.savefig(os.path.join(snapshotdir, f'snapshot_{angle+45}_{lyr}.{filetype}'), **sf_kwargs)

    plt.close(fig)

    if not video:
        return

    scenes = [
        RotationScene(INPUT_SPIN_TIME, layer_names[0] if layer_names is not None else '', 0)
    ]

    if clusts[0] is not None:
        clust: clusters.Clusters = clusts[0]
        for clust_ind in range(clust.num_clusters):
            scenes.append(_get_cluster_scene(
                traj,
                0,
                layer_names[0] if layer_names is not None else '',
                clust,
                clust_ind))

    curdat = traj.snapshots[0].projected_samples.numpy()
    curlim = (float(curdat.min()), float(curdat.max()))
    for i in range(1, traj.num_layers):
        interptitle = f'{layer_names[i - 1]} -> {layer_names[i]}' if layer_names is not None else ''

        newdat = traj.snapshots[i].projected_samples.numpy()
        newlim = (float(newdat.min()), float(newdat.max()))
        interplim = (min(curlim[0], newlim[0]), max(curlim[1], newlim[1]))
        if interplim[0] != curlim[0] or interplim[1] != curlim[1]:
            scenes.append(ZoomScene(ZOOM_TIME, interptitle, curlim, interplim, i-1))
        scenes.append(InterpScene(
            INTERP_SPIN_TIME,
            interptitle,
            i - 1,
            i,
        ))
        if interplim[0] != newlim[0] or interplim[1] != newlim[1]:
            scenes.append(ZoomScene(ZOOM_TIME, interptitle, interplim, newlim, i))

        scenes.append(RotationScene(
            OTHER_SPIN_TIME,
            layer_names[i] if layer_names is not None else '',
            i
        ))
        if clusts[i] is not None:
            clust: clusters.Clusters = clusts[i]
            for clust_ind in range(clust.num_clusters):
                scenes.append(_get_cluster_scene(
                    traj,
                    i,
                    layer_names[i] if layer_names is not None else '',
                    clust,
                    clust_ind))
        curlim = newlim
        curdat = newdat

    sum_time = sum((scene.duration for scene in scenes), 0)
    num_frames = int(sum_time / frame_time)

    animator = saa.MPAnimation(DPI, (FRAME_SIZE[0] * DPI, FRAME_SIZE[1] * DPI), fps,
                               os.path.join(outfile_wo_ext, 'out.mp4'),
                               os.path.join(outfile_wo_ext, 'mp_anim.log'))

    workers = []
    for i in range(NUM_WORKERS):
        wlog = os.path.join(outfile_wo_ext, f'worker_{i}_error.log')
        img_queue = myq.ZeroMQQueue.create_recieve()
        send_queue = myq.ZeroMQQueue.create_send()
        ack_queue = myq.ZeroMQQueue.create_recieve()
        if isinstance(traj, pca_ff.PCTrajectoryFF):
            proc = Process(
                target=_frame_worker_target,
                args=(img_queue.serd(), send_queue.serd(), ack_queue.serd(), frame_time,
                      FRAME_SIZE, DPI, scenes, traj, s, wlog))
        else:
            proc = Process(
                target=_frame_worker_target_gen,
                args=(img_queue.serd(), send_queue.serd(), ack_queue.serd(), frame_time,
                      FRAME_SIZE, DPI, scenes, traj, s, markers, scalar_mapping, norm, cmap, wlog))
        proc.start()
        workers.append(FrameWorkerConnection(proc, img_queue, send_queue, ack_queue))
        animator.register_queue(img_queue)

    animator.start()

    last_print = time_.time()
    last_frame = 0
    for i in range(0, num_frames, len(workers)):
        sync_reqd = (i % FRAMES_PER_SYNC) == 0
        if sync_reqd:
            for worker in workers:
                worker.start_sync()

            while True:
                animator.do_work()

                worker_not_done = False
                for worker in workers:
                    if not worker.check_sync():
                        worker_not_done = True
                        break
                if not worker_not_done:
                    break

        for worker_ind, worker in enumerate(workers):
            if i + worker_ind < num_frames:
                worker.send(i + worker_ind)

        if time_.time() - last_print > PRINT_EVERY:
            deltime = time_.time() - last_print
            last_print = time_.time()
            delframes = i - last_frame
            last_frame = i
            frames_per_second = delframes / deltime
            framesleft = num_frames - i
            timeremaining = framesleft / frames_per_second
            print(f'[PCA3D] {(i/num_frames)*100:.2f}% ({timeremaining:.0f}s rem) ({i}/{num_frames}) Did {delframes} in last {deltime:.1f}s ({frames_per_second:.1f} frames/sec)')
            sys.stdout.flush()

        animator.do_work()
        while len(animator.ooo_frames) > 100:
            animator.do_work() # getting behind

    for worker in workers:
        worker.finish()

    animator.finish()

    if os.path.exists(outfile):
        os.remove(outfile)
    zipdir(outfile_wo_ext)

def plot_ff(traj: pca_ff.PCTrajectoryFF, outfile: str, exist_ok: bool,
            frame_time: float = 16.67, digestor: NPDigestor = None,
            layer_names: typing.List[str] = None):
    """Plots the given trajectory to the given outfile if possible. If the
    digestor is given, then this effect takes place on a different thread

    Arguments:
        traj (pca_ff.PCTrajectoryFF): the trajectory to plot
        outfile (str): where to plot the trajectory
        exist_ok (bool): True to overwrite existing plot, false to error if it exists
        digestor (NPDigestor, optional): Default None. If specified, this will be used
            to multiprocess the operation
        layer_names (list[str], optional): Default None. If specified, this will be the title
            when the corresponding layer is visible
    """
    if not isinstance(traj, pca_ff.PCTrajectoryFF):
        raise ValueError(f'expected traj is PCTrajectoryFF, got {traj} (type={type(traj)})')
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok} (type={type(exist_ok)})')
    if digestor is not None and not callable(digestor):
        raise ValueError(f'expected digestor is callable, got {digestor} (type={type(digestor)})')
    if layer_names is not None:
        if not isinstance(layer_names, (list, tuple)):
            raise ValueError(f'expected layer_names is list, got {layer_names} (type={type(layer_names)})')
        for idx, val in enumerate(layer_names):
            if not isinstance(val, str):
                raise ValueError(f'expected layer_names[{idx}] is str, got {val} (type={type(val)})')
        if len(layer_names) != traj.num_layers:
            raise ValueError(f'expected len(layer_names) = traj.num_layers = {traj.num_layers} but is {len(layer_names)}')

    outfile_wo_ext = os.path.splitext(outfile)[0]
    if outfile == outfile_wo_ext:
        outfile += '.zip'

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'to write {outfile} need {outfile_wo_ext} available')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'outfile {outfile} already exists (use exist_ok=True to overwrite)')

    if digestor is None:
        _plot_ff_real(traj, outfile, exist_ok, frame_time=frame_time, layer_names=layer_names)
        return

    sample_labels = traj.snapshots[0].projected_sample_labels.numpy()
    args = []
    for snapshot in traj.snapshots:
        snapshot: pca_ff.PCTrajectoryFFSnapshot
        args.append(snapshot.principal_vectors.numpy())
        args.append(snapshot.principal_values.numpy())
        args.append(snapshot.projected_samples.numpy())
    digestor(sample_labels, *args, outfile=outfile, exist_ok=exist_ok,
             frame_time=frame_time, layer_names=layer_names,
             target_module='shared.measures.pca_3d', target_name='_plot_npmp')

def plot_gen(traj: pca_gen.PCTrajectoryGen, outfile: str, exist_ok: bool,
             markers: str, scalar_mapping: str, norm: str,
             cmap: str = 'cividis', frame_time: float = 16.67, s: float = 1,
             digestor: NPDigestor = None, layer_names: typing.List[str] = None,
             clusts: typing.Optional[typing.List[typing.Optional[clusters.Clusters]]] = None):
    """Plots the given general trajectory in 3-dimensions in a smooth video,
    optionally using the given digestor. Because the general trajectories do
    not have any assumptions about the output layer, more information is needed
    to know how to plot them.

    Args:
        traj (pca_gen.PCTrajectoryGen): The trajectory to plot
        outfile (str): Where to store the video and snapshots
        exist_ok (bool): True if we should delete existing folders that we want to use if
            they exist, False to error if there are existing folders we want to use.
        markers (str): A path to a callable that returns a callable which accepts
            (ndarray[batch_size, output_size]) and returns (ndarray[batch_size], str) where the
            output tensor is a mask for the samples and the output string is the marker to use.
        scalar_mapping (str): A path to something that, when called, gives an OutputToScalarMapping
        norm (str): A path to something that, when called, gives the matplotlib.colors.Normalize
            instance to use
        cmap (str, optional): The color map to use. Defaults to 'cividis'.
        frame_time (float, optional): Number of milliseconds per frame. Defaults to 16.67.
        digestor (NPDigestor, optional): If provided, used to digest. Defaults to None.
        layer_names (typing.List[str], optional): If provided, used to create titles for scenes.
            Defaults to None.
    """
    tus.check(
        traj=(traj, pca_gen.PCTrajectoryGen),
        outfile=(outfile, str),
        exist_ok=(exist_ok, bool),
        markers=(markers, str),
        scalar_mapping=(scalar_mapping, str),
        norm=(norm, str),
        cmap=(cmap, str),
        frame_time=(frame_time, float),
        s=(s, (int, float))
    )
    mutils.process_outfile(outfile, exist_ok)

    if digestor is None:
        _plot_ff_real(traj, outfile, exist_ok, frame_time, layer_names,
                      None, True, markers, scalar_mapping, norm, cmap, s, clusts)
        return
    elif clusts is not None:
        raise NotImplementedError('passing clusters to digestor')

    sample_labels = traj.snapshots[0].projected_sample_labels.numpy()
    args = []
    for snapshot in traj.snapshots:
        snapshot: pca_gen.PCTrajectoryGenSnapshot
        args.append(snapshot.principal_vectors.numpy())
        args.append(snapshot.principal_values.numpy())
        args.append(snapshot.projected_samples.numpy())
    digestor(sample_labels, *args, outfile=outfile, exist_ok=exist_ok,
             frame_time=frame_time, layer_names=layer_names,
             markers=markers, scalar_mapping=scalar_mapping, norm=norm, cmap=cmap, s=s,
             target_module='shared.measures.pca_3d', target_name='_plot_npmp')


def during_training(savepath: str, train: bool, digestor: typing.Optional[NPDigestor] = None,
                    frame_time: float = 16.67, plot_kwargs: dict = None):
    """Fetches the on_step/on_epoch for things like OnEpochsCaller
    that saves into the given directory.

    Args:
        savepath (str): where to save
        train (bool): true to use training data, false to use validation data
        digestor (NPDigestor, optional): if specified, used for multiprocessing
        frame_time (float, optional): the milliseconds per frame
        plot_kwargs (dict, optional): passed onto the plotter
    """
    if not isinstance(savepath, str):
        raise ValueError(f'expected savepath is str, got {savepath} (type={type(savepath)})')
    if not isinstance(train, bool):
        raise ValueError(f'expected train is bool, got {train} (type={type(train)})')
    if digestor is not None and not isinstance(digestor, NPDigestor):
        raise ValueError(f'expected digestor is NPDigestor, got {digestor} (type={type(digestor)})')

    if os.path.exists(savepath):
        raise ValueError(f'{savepath} already exists')

    def on_step(context: GenericTrainingContext, fname_hint: str):
        context.logger.info('[PCA_FF] Measuring PCA (3D) Through Layers (hint: %s)', fname_hint)
        pwl = context.train_pwl if train else context.test_pwl
        outfile = os.path.join(savepath, f'pca_{fname_hint}')
        traj = pca_ff.find_trajectory(context.model, pwl, 3)
        plot_ff(traj, outfile, False, frame_time, digestor, **plot_kwargs)

    return on_step

