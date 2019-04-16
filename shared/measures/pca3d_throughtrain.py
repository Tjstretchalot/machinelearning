"""This module produces an animated 3d pc-plot through training. It outputs 1 video
per layer.

The raw arrays are sent via memory sharing.
"""

import typing
import os
from multiprocessing import Queue, Process
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shared.trainer import GenericTrainingContext
from shared.filetools import zipdir
import shared.measures.utils as mutils
from shared.async_anim import AsyncAnimation
import shared.mytweening as mytweening
import shared.measures.pca_ff as pca_ff
import shared.measures.pca as pca

MESSAGE_STYLES = {'start', 'hidacts', 'hidacts_done', 'videos_done'}
"""
start:
    outpath (str): the path to the folder to save animations in
    batch_size (int): the batch size
    layer_names (tuple[str]): the name of each layer as you will see it through
        the network, starting with 'input' and ending on 'output'. there are
        len(layer_names) - 1 layers in the network in total
    layer_sizes (tuple[int]): the number of hidden nodes that will be visible in
        each layer. the first value is the input dim the last value is the output
        dim
    sample_labels_file (str): the file that we take sample_labels form
    hid_acts_files (tuple[str]): the files that contain the hidden activations, with
        the shape (batch_size, layer_size)

    layer (int) which layer (where 0 is input) that this worker is responsible for
hidacts:
    sent from the main thread to the worker thread to tell it that the hidden
    activations have been updated, sent from the worker thread to the main thread
    to tell it we are ready for another one.

    epoch (int): which epoch we're in
hidacts_done:
    sent from main thread to worker thread to tell it to finish
videos_done:
    sent from worker thread to main thread to confirm video finished
"""

FRAMES_PER_TRAIN = 2
MS_PER_ROTATION = 3000
ROTATION_EASING = mytweening.smoothstep
FRAME_TIME = 1000 / 60.

class Worker:
    """Describes a worker instance. A worker manages rendering of a single layer.

    Attributes:
        receive_queue (Queue): the queue we receive messages from
        send_queue (Queue): the queue we send messages through
    """

    def __init__(self, receive_queue: Queue, send_queue: Queue):
        self.receive_queue = receive_queue
        self.send_queue = send_queue

    def work(self):
        """Processes videos"""
        start_msg = self.receive_queue.get()
        if start_msg[0] != 'start':
            raise ValueError(f'expected \'start\' message, got {start_msg}')

        data = start_msg[1]
        outpath = data['outpath']
        batch_size = data['batch_size']
        layer_names = data['layer_names']
        if layer_names[0] != 'input':
            raise ValueError(f'expected layer_names[0] is input, but is {layer_names[0]}')
        if layer_names[-1] != 'output':
            raise ValueError(f'expected layer_names[-1] is output, but is {layer_names[-1]}')
        layer_sizes = data['layer_sizes']
        if len(layer_names) != len(layer_sizes):
            raise ValueError(f'expected len(layer_names) = len(layer_sizes); got {len(layer_names)} layer names and {len(layer_sizes)} layer sizes')

        sample_labels_file = data['sample_labels_file']
        hid_acts_files = data['hid_acts_files']
        if len(hid_acts_files) != len(layer_names):
            raise ValueError(f'expected len(layer_names) = len(hid_acts_files); got {len(layer_names)} layer names and {len(hid_acts_files)} hid act files')

        layer_idx = data['layer']

        sample_labels = np.memmap(sample_labels_file, dtype='int32', mode='r', shape=(batch_size,))
        sample_labels_torch = torch.from_numpy(sample_labels)
        layer_size = layer_sizes[layer_idx]
        layer_data = np.memmap(hid_acts_files[layer_idx], dtype='float64', mode='r', shape=(batch_size, layer_size))
        layer_name = layer_names[layer_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        anim = AsyncAnimation(fig)
        anim.prepare_save(os.path.join(outpath, f'layer_{layer_idx}_{layer_names[layer_idx]}'), fps=60, dpi=100)

        msg = self.receive_queue.get()

        figdata = self._init_frame(fig, ax, layer_data, layer_name, sample_labels_torch)

        self.send_queue.put(('hidacts',))
        msg = self.receive_queue.get()
        while msg[0] == 'hidacts':
            self._do_frames(
                anim, fig, ax, layer_data,
                msg[1]['epoch'], figdata)
            self.send_queue.put(('hidacts',))
            msg = self.receive_queue.get()

        if msg[0] != 'hidacts_done':
            raise ValueError(f'expected msg[0] = \'hidacts_done\' but is {msg[0]}')

        self._do_finish(anim, fig, ax, figdata)
        anim.on_finish()

        self.send_queue.put(('videos_done',))
        while not self.send_queue.empty():
            time.sleep(0.01)

        sample_labels._mmap.close() # pylint: disable=protected-access
        layer_data._mmap.close() # pylint: disable=protected-access

    def _make_snapshot(self, hid_acts: np.ndarray, sample_labels_torch: torch.tensor):
        torch_hidacts = torch.from_numpy(hid_acts)
        pc_vals, pc_vecs = pca.get_hidden_pcs(torch_hidacts, 3)
        projected = pca.project_to_pcs(torch_hidacts, pc_vecs, out=None)
        return pca_ff.PCTrajectoryFFSnapshot(pc_vecs, pc_vals, projected, sample_labels_torch)

    def _init_frame(self, fig: mpl.figure.Figure, ax: mpl.axes.Axes, hid_acts: np.ndarray,
                    layer_name: str, sample_labels_torch: torch.tensor) -> dict:
        snapsh = self._make_snapshot(hid_acts, sample_labels_torch)
        scatter = ax.scatter(snapsh.projected_samples[:, 0].numpy(),
                             snapsh.projected_samples[:, 1].numpy(),
                             snapsh.projected_samples[:, 2].numpy(),
                             s=1,
                             c=snapsh.projected_sample_labels.numpy())
        axtitle = ax.set_title(layer_name)
        return {
            'layer_name': layer_name,
            'scatter': scatter,
            'title': axtitle,
            'time': 0,
            'sample_labels_torch': sample_labels_torch
        }

    def _do_frames(self, anim: AsyncAnimation, fig: mpl.figure.Figure, ax: mpl.axes.Axes,
                   hid_acts: np.ndarray, epoch: int, figdata: dict) -> float:

        snapsh = self._make_snapshot(hid_acts, figdata['sample_labels_torch'])
        figdata['scatter']._offsets3d = ( #pylint: disable=protected-access
            snapsh.projected_samples[:, 0].numpy(),
            snapsh.projected_samples[:, 1].numpy(),
            snapsh.projected_samples[:, 2].numpy())
        ax.set_xlim(float(snapsh.projected_samples[:, 0].min()), float(snapsh.projected_samples[:, 0].max()))
        ax.set_ylim(float(snapsh.projected_samples[:, 1].min()), float(snapsh.projected_samples[:, 1].max()))
        ax.set_zlim(float(snapsh.projected_samples[:, 2].min()), float(snapsh.projected_samples[:, 2].max()))
        figdata['title'].set_text(figdata['layer_name'] + f': epoch {epoch}')

        for _ in range(FRAMES_PER_TRAIN):
            prog = figdata['time'] / MS_PER_ROTATION
            rotation = float(45 + 360 * np.sin(ROTATION_EASING(prog) * np.pi * 2))
            ax.view_init(30, rotation)
            anim.on_frame((ax, figdata['title'], figdata['scatter']))

            figdata['time'] += FRAME_TIME
            if figdata['time'] > MS_PER_ROTATION:
                figdata['time'] -= MS_PER_ROTATION

    def _do_finish(self, anim: AsyncAnimation, fig: mpl.figure.Figure, ax: mpl.axes.Axes,
                   figdata: dict):
        figdata['title'].set_text(figdata['layer_name'] + ': finished')
        start_time = figdata['time']
        ms_finish_time = (MS_PER_ROTATION - (start_time % MS_PER_ROTATION)) + MS_PER_ROTATION
        time_so_far = 0
        while time_so_far < ms_finish_time:
            cur_time = (start_time + time_so_far) % MS_PER_ROTATION
            prog = cur_time / MS_PER_ROTATION
            rotation = float(45 + 360 * np.sin(ROTATION_EASING(prog) * np.pi * 2))
            ax.view_init(30, rotation)

            if time_so_far == 0:
                anim.on_frame((ax, figdata['title'],))
            else:
                anim.on_frame((ax,))

            time_so_far += FRAME_TIME


def _worker_target(receive_queue, send_queue):
    worker = Worker(receive_queue, send_queue)
    worker.work()

class WorkerConnection:
    """Describes the main threads connection to the worker.

    Attributes:
        process (Process): the worker process
        send_queue (Queue): the main thread to worker queue
        receive_queue (Queue): the worker to main thread queue

        expecting_hidacts_ack (bool): True if we are expecting the worker
            to acknowledge a hidacts, False otherwise
    """

    def __init__(self, process: Process, send_queue: Queue, receive_queue: Queue):
        self.process = process
        self.send_queue = send_queue
        self.receive_queue = receive_queue
        self.expecting_hidacts_ack = False

    def check_ack(self):
        """Fetches a hidacts acknowledge message from the receive queue if appropriate"""
        if self.expecting_hidacts_ack:
            msg = self.receive_queue.get()
            if msg[0] != 'hidacts':
                raise ValueError(f'expected hidacts response, got {msg}')

            self.expecting_hidacts_ack = False

    def send_hidacts(self, epoch: int):
        """Sends the 'hidacts' message to the worker"""
        self.check_ack()
        self.send_queue.put(('hidacts', {'epoch': epoch}))
        self.expecting_hidacts_ack = True

    def start_finish(self):
        """Shuts down the worker gracefully after all hidacts have been sent"""
        self.check_ack()
        self.send_queue.put(('hidacts_done',))

    def end_finish(self):
        """Should be called after start_finish to wait until the worker shutdown"""
        msg = self.receive_queue.get()
        if msg[0] != 'videos_done':
            raise ValueError(f'expected videos_done acknowledge')
        while self.process.is_alive():
            time.sleep(0.01)

class PCAThroughTrain:
    """This is setup to be added to the GenericTrainer directly. This will spawn
    workers which will manage piping the data to all the ffmpeg instances that are spawned.

    Attributes:
        output_folder (str): the folder that we are outputting data into. The folder will be
            archived once the files are ready. We will also use this folder to share information
            with the worker thread via memmap'd files
        layer_names (list[str]): the name of each layer starting with 'input' and ending with
            'output'

        connections (list[WorkerConnection]): the connections to the workers

        batch_size (int): the number of points we are plotting
        sample_labels [np.ndarray]: the memmap'd int32 array we share labels with
        sample_points [np.ndarray]: the unmapped float64 array we store the points in

        layers [list[np.ndarray]]: the list of memmap'd float64 arrays we share hid_acts with

        sample_labels_torch [torch.tensor]: torch.from_numpy(sample_labels)
        sample_points_torch [torch.tensor]: torch.from_numpy(sample_points)

        sample_labels_file (str): the path to the sample labels mmap'd file
        hid_acts_files (list[str]): the paths to the hidden activation files (by layer) mmap'd
    """

    def __init__(self, output_path: str, layer_names: typing.List[str], exist_ok: bool = False):
        """
        Args:
            output_path (str): either the output folder or the output archive
            layer_names (str): a list of layer names starting with 'input' and
                ending with 'output'
            exist_ok (bool, optional): Defaults to False. If True, if the output
                archive already exists it will be deleted. Otherwise, if the output
                archive already exists an error will be raised
        """

        _, self.output_folder = mutils.process_outfile(output_path, exist_ok)
        self.layer_names = layer_names

        self.connections = None

        self.batch_size = None
        self.sample_labels = None
        self.sample_points = None
        self.layers = None

        self.sample_labels_torch = None
        self.sample_points_torch = None

        self.sample_labels_file = None
        self.hid_acts_files = None

    def setup(self, context: GenericTrainingContext, **kwargs):
        """Spawns the worker"""
        os.makedirs(self.output_folder)
        self.batch_size = min(100 * context.test_pwl.output_dim, context.test_pwl.epoch_size)

        self.sample_labels_file = os.path.join(self.output_folder, 'sample_labels.bin')
        self.sample_labels = np.memmap(self.sample_labels_file, dtype='int32',
                                       mode='w+', shape=(self.batch_size,))

        self.sample_points = np.zeros((self.batch_size, context.test_pwl.input_dim), dtype='float64')

        self.sample_labels_torch = torch.from_numpy(self.sample_labels)
        self.sample_points_torch = torch.from_numpy(self.sample_points)

        context.test_pwl.mark()
        context.test_pwl.fill(self.sample_points_torch, self.sample_labels_torch)
        context.test_pwl.reset()

        acts = mutils.get_hidacts_with_sample(context.model, self.sample_points_torch, self.sample_labels_torch)

        layer_sizes = []
        self.hid_acts_files = []
        self.layers = []
        for idx, lyr in enumerate(acts.hid_acts):
            filepath = os.path.join(self.output_folder, f'hid_acts_{idx}.bin')
            self.hid_acts_files.append(filepath)
            layer_sizes.append(int(lyr.shape[1]))
            self.layers.append(np.memmap(filepath, dtype='float64', mode='w+', shape=(self.batch_size, int(lyr.shape[1]))))

        self.connections = []
        for lyr in range(len(self.layers)):
            send_queue = Queue()
            receive_queue = Queue()
            proc = Process(target=_worker_target, args=(send_queue, receive_queue)) # swapped
            proc.daemon = True
            proc.start()

            send_queue.put((
                'start',
                {
                    'outpath': self.output_folder,
                    'batch_size': self.batch_size,
                    'layer_names': self.layer_names,
                    'layer_sizes': layer_sizes,
                    'sample_labels_file': self.sample_labels_file,
                    'hid_acts_files': tuple(self.hid_acts_files),
                    'layer': lyr
                }
            ))
            connection = WorkerConnection(proc, send_queue, receive_queue)
            self.connections.append(connection)

    def _send_hidacts(self, context: GenericTrainingContext):
        """Runs sample_points through the network and sends those activations to
        the worker thread.
        """

        nhacts = mutils.get_hidacts_with_sample(
            context.model, self.sample_points_torch, self.sample_labels_torch)
        for idx, lyr in enumerate(nhacts.hid_acts):
            self.layers[idx][:] = lyr
        for connection in self.connections:
            connection.send_hidacts(int(context.shared['epochs'].epochs))

    def pre_loop(self, context: GenericTrainingContext):
        """Feeds hidden activations to the network"""
        self._send_hidacts(context)

    def finish(self, context: GenericTrainingContext):
        """Finishes the worker, closes and deletes mmap'd files, zips directory"""
        context.logger.info('[PCA3D-ThroughTrain] Cleaning up and archiving')
        self._send_hidacts(context)

        for connection in self.connections:
            connection.start_finish()
        for connection in self.connections:
            connection.end_finish()
        self.connections = None

        self.sample_labels_torch = None
        self.sample_points_torch = None

        self.sample_labels._mmap.close()
        self.sample_labels = None

        self.sample_points = None

        for lyr in self.layers:
            lyr._mmap.close()

        self.layers = None

        os.remove(self.sample_labels_file)

        for hafile in self.hid_acts_files:
            os.remove(hafile)

        self.sample_labels_file = None
        self.hid_acts_files = None

        zipdir(self.output_folder)








