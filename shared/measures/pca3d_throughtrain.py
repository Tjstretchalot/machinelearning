"""This module produces an animated 3d pc-plot through training. It outputs 1 video
per layer.

The raw arrays are sent via memory sharing.
"""

import typing
import os
from multiprocessing import Queue, Process
import time
import queue
import io

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shared.trainer import GenericTrainingContext
from shared.filetools import zipdir
import shared.measures.utils as mutils
from shared.async_anim import MPAnimation
import shared.mytweening as mytweening
import shared.measures.pca_ff as pca_ff
import shared.measures.pca as pca

MESSAGE_STYLES = {'start', 'hidacts', 'hidacts_done', 'videos_done'}
"""
These are messages that are sent from the main thread to the layer worker thread

start:
    worker_id (str): a unique id for the worker
    outfile (str): the path to the file where the animation should be saved
    batch_size (int): the batch size
    layer_name (str): the name of the layer the layer worker is handling
    layer_size (int): the number of hidden nodes that will be visible in the layer
        that this layer worker is handling
    output_dim (int): the number of labels / output embedded dim
    sample_labels_file (str): the file that we take sample_labels form
    hid_acts_file (str): the file that contain the hidden activations, with
        the shape (batch_size, layer_size)
    num_frame_workers (int): how many frame workers to use
    dpi (int or float): number of pixels per inch
    fps (int): number of frames per second
    frame_size (tuple[float, float]) width and height of frame in inches
    ffmpeg_logfile (str): where the ffmpeg logs are stored
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

FRAMES_PER_TRAIN = 4
MS_PER_ROTATION = 10000
ROTATION_EASING = mytweening.smootheststep
FPS = 60
FRAME_TIME = 1000.0 / FPS
FRAME_SIZE = (19.2, 10.8) # oh baby
DPI = 100

class FrameWorker:
    """Describes the frame worker instance. This manages rendering individual frames that will
    go into the overall animation. We can get a benefit out of having more than one of these
    because we will render more than one frame per training sample.

    Attributes:
        receive_queue (Queue): the queue we receive messages from. Messages come in the form of
            either ('frame', rotation, title, index) or ('end',)
        img_queue (Queue): the queue we send messages in the form (index, rawdata) where rawdata
            is the bytes that ought to be sent to the ffmpeg process.
        ack_queue (Queue): the queue we send messages in the form ('ack',) once we have completed
            a frame. this lets the layer worker know its safe to tell the main thread that the
            hidden activations file can be modified
        ack_mode (str): determines when we send the 'ack' message through the ack_queue
                'asap': we send as soon as modifying hidden_acts would not hurt our rendering
                'ready': we send as soon as we're ready for a new hidden_acts

        hacts_file (str): the path to the hidden activations file, which contains the memmappable
            [batch_size x layer_size] data
        labels_file (str): the path to the labels file which contains the [batch_size] data.
            we only need to load this temporarily for the purpose of blitting the colors and
            from there on we only need to rotate images, however we memmap in order to use
            the pca calculations more smoothly
        match_mean_comps_file (str): the path to the file which will contain the 'mean_comps' that
            we should match our snapshots to; updated occassionally via a 'match' call and ensures
            deterministic behavior among frame workers

        batch_size (int): the batch size
        layer_size (int): the layer size
        output_dim (int): the output dim / number of labels

        hidden_acts (np.memmap): the memory mapped hidden activations
        hidden_acts_torch (torch.tensor): the torch tensor that has the same backing as hidden_acts

        sample_labels (np.memmap): the memory mapped sample labels
        sample_labels_torch (torch.tensor): the torch tensor that has the same backing as sample_labels

        match_mean_comps (np.memmap): the memory mapped "mean_comps" for the snapshot match info
        match_mean_comps_torch (torch.tensor): the torch.tensor that has the same backing
            as match_mean_comps

        match_info (pca_ff.PCTrajectoryFFSnapshotMatchInfo): the match info we use to keep
            the snapshots within reflection distance of each other

        rotation (float): the rotation around the y axes we should have
        title (str): the title we should have
        index (int): the index we will send to the img queue
        snapshot (pca_ff.PCTrajectoryFFSnapshot): the snapshot that we ought to plot

        figure (mpl.figure.Figure): the figure that we are rendering onto
        axes (mpl.axes.Axes): the 3d axes that we are scattering onto
        axtitle (any): the Text object that we can set_text to change the title
        scatter (any): the 3d scatter plot we can modify

        frame_format (str): the frame format we use
        w_in (float): width in inches
        h_in (float): height in inches
        dpi (int): the dpi we use

        last_job (float): the last time we received a job

        state (int):
            0 = haven't started yet
            1 = awaiting receive_queue message
            2 = calculated trajectory
            3 = finished
    """



    def __init__(self, receive_queue: Queue, img_queue: Queue, ack_queue: Queue, ack_mode: str,
                 hacts_file: str, labels_file: str, match_mean_comps_file: str, batch_size: int,
                 layer_size: int, output_dim: int, w_in: float, h_in: float, dpi: int):
        self.receive_queue = receive_queue
        self.img_queue = img_queue
        self.ack_queue = ack_queue
        self.ack_mode = ack_mode
        self.hacts_file = hacts_file
        self.labels_file = labels_file
        self.match_mean_comps_file = match_mean_comps_file
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.output_dim = output_dim
        self.hidden_acts = None
        self.hidden_acts_torch = None
        self.sample_labels = None
        self.sample_labels_torch = None
        self.match_mean_comps = None
        self.match_mean_comps_torch = None
        self.match_info = None
        self.rotation = None
        self.title = None
        self.index = None
        self.snapshot = None
        self.figure = None
        self.axes = None
        self.axtitle = None
        self.scatter = None

        self.frame_format = 'rgba' #str(mpl.rcParams['animation.frame_format'])
        self.w_in = w_in
        self.h_in = h_in
        self.dpi = dpi

        self.last_job = time.time()
        self.state = 0

    def _open_mmaps(self):
        self.hidden_acts = np.memmap(self.hacts_file, dtype='float64',
                                     mode='r', shape=(self.batch_size, self.layer_size))
        self.hidden_acts_torch = torch.from_numpy(self.hidden_acts)

        self.sample_labels = np.memmap(self.labels_file, dtype='int32',
                                       mode='r', shape=(self.batch_size,))
        self.sample_labels_torch = torch.from_numpy(self.sample_labels)
        self.match_mean_comps = np.memmap(
            self.match_mean_comps_file, dtype='uint8', mode='r',
            shape=(pca_ff.PCTrajectoryFFSnapshotMatchInfo.get_expected_len(3, self.output_dim)))
        self.match_mean_comps_torch = torch.from_numpy(self.match_mean_comps)

        self.match_info = pca_ff.PCTrajectoryFFSnapshotMatchInfo(
            3, self.layer_size, self.output_dim, self.match_mean_comps_torch)

    def _close_mmaps(self):
        self.hidden_acts_torch = None
        self.hidden_acts._mmap.close() # pylint: disable=protected-access
        self.hidden_acts = None

        self.sample_labels_torch = None
        self.sample_labels._mmap.close() # pylint: disable=protected-access
        self.sample_labels = None

        self.match_info = None
        self.match_mean_comps_torch = None
        self.match_mean_comps._mmap.close() # pylint: disable=protected-access
        self.match_mean_comps = None

    def _get_snapshot(self):
        pc_vals, pc_vecs = pca.get_hidden_pcs(self.hidden_acts_torch, 3)
        projected = pca.project_to_pcs(self.hidden_acts_torch, pc_vecs, out=None)
        snap = pca_ff.PCTrajectoryFFSnapshot(pc_vecs, pc_vals, projected, self.sample_labels_torch)

        self.match_info.match(snap)

        self.snapshot = snap

    def _init_figure(self):
        if self.snapshot is None:
            raise ValueError(f'_init_figure requires a snapshot')

        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111, projection='3d')

        self.scatter = self.axes.scatter(
            self.snapshot.projected_samples[:, 0].numpy(),
            self.snapshot.projected_samples[:, 1].numpy(),
            self.snapshot.projected_samples[:, 2].numpy(),
            s=3,
            c=self.snapshot.projected_sample_labels.numpy())
        self.axtitle = self.axes.set_title(self.title)

        self.axes.view_init(30, self.rotation)

    def _close_figure(self):
        self.axtitle = None
        self.scatter = None
        self.axes = None
        self.figure = None

    def _setup_frame(self):
        snapsh = self.snapshot
        self.scatter._offsets3d = (snapsh.projected_samples[:, 0].numpy(), # pylint: disable=protected-access
                                   snapsh.projected_samples[:, 1].numpy(),
                                   snapsh.projected_samples[:, 2].numpy())
        self.axes.set_xlim(float(snapsh.projected_samples[:, 0].min()),
                           float(snapsh.projected_samples[:, 0].max()))
        self.axes.set_ylim(float(snapsh.projected_samples[:, 1].min()),
                           float(snapsh.projected_samples[:, 1].max()))
        self.axes.set_zlim(float(snapsh.projected_samples[:, 2].min()),
                           float(snapsh.projected_samples[:, 2].max()))
        self.axes.view_init(30, self.rotation)
        self.axtitle.set_text(self.title)

    def _create_frame(self) -> bytes:
        hndl = io.BytesIO()
        self.figure.set_size_inches(self.w_in, self.h_in)
        self.figure.savefig(hndl, format=self.frame_format, dpi=self.dpi)
        return hndl.getvalue()

    def work(self):
        """Initializes and produces frames until the end message is received"""

        while self.do_work():
            pass

    def do_work(self, no_wait=False) -> bool:
        """Performs a bit of work until we have no more work to do

        Returns:
            have_work (bool): True if there is work to do, False otherwise
        """

        if self.state == 0:
            self._open_mmaps()
            self.state = 1
            return True
        if self.state == 1:
            if no_wait and self.receive_queue.empty():
                return True
            msg = self.receive_queue.get() # a 15 second timeout gets hit here? I think side effect of too many workers?
            self.last_job = time.time()
            if msg[0] == 'end':
                self._close_mmaps()
                self._close_figure()
                self.state = 3
                return False

            self.rotation, self.title, self.index = msg[1:]
            self.state = 2
            return True
        if self.state == 2:
            self._get_snapshot()
            if self.ack_mode == 'asap':
                self.ack_queue.put(('ack',))
            if self.figure is None:
                self._init_figure()
            self._setup_frame()
            frm = self._create_frame()
            self.img_queue.put((self.index, frm))
            if self.ack_mode == 'ready':
                self.ack_queue.put(('ack',))
            self.state = 1
            return True

        return False

def _frame_worker_target(receive_queue: Queue, img_queue: Queue, ack_queue: Queue, ack_mode: str,
                         hacts_file: str, labels_file: str, match_means_comp_file: str,
                         batch_size: int, layer_size: int, output_dim: int, w_in: float,
                         h_in: float, dpi: int):
    worker = FrameWorker(receive_queue, img_queue, ack_queue, ack_mode, hacts_file, labels_file,
                         match_means_comp_file, batch_size, layer_size, output_dim, w_in, h_in, dpi)
    worker.work()

class FrameWorkerConnection:
    """Describes the connection a LayerWorker has with a FrameWorker

    Attributes:
        process (Process): the process that the frame worker is running on
        job_queue (Queue): the queue which we send jobs to
        ack_queue (Queue): the queue which we receive acks from

        awaiting_ack (bool): True if we haven't seen an ack from the last job yet
    """

    def __init__(self, process: Process, job_queue: Queue, ack_queue: Queue):
        self.process = process
        self.job_queue = job_queue
        self.ack_queue = ack_queue
        self.awaiting_ack = False

    def check_ack(self):
        """Non-blocking equivalent of wait_ack"""
        if self.awaiting_ack and not self.ack_queue.empty():
            ack = self.ack_queue.get_nowait()
            if ack is not None:
                self.awaiting_ack = False

    def wait_ack(self):
        """Waits for the acknowledgement of the job from the frame worker"""
        if self.awaiting_ack:
            self.ack_queue.get(timeout=15)
            self.awaiting_ack = False

    def send_job(self, rotation: float, title: str, index: int):
        """Tells the frame worker to produce an image of the given rotation and title. The
        index is the index of the frame within the video, where 0 is the first frame.

        Arguments:
            rotation (float): in degrees around the y axis
            title (str): the title of the image
            index (int): the index in the video of the frame
        """
        self.wait_ack()
        self.job_queue.put(('frame', rotation, title, index))
        self.awaiting_ack = True

    def send_end(self):
        """Tells the worker to shutdown"""
        self.job_queue.put(('end',))

    def wait_end(self):
        """Waits for the worker to shutdown"""
        self.process.join()

    def check_end(self) -> bool:
        """Checks if the worker has shutdown yet"""
        return not self.process.is_alive()

class LayerWorker:
    """Describes a worker instance. A worker manages rendering of a single layer. It itself
    will delegate to FrameWorkers which will render the individual frames.

    Example training a 7-layer network: the main thread delegates to 7 layer thread which each
        delegate to 4 frame threads. A total of 30 threads are used for generating this plot,
        which is probably only going to work on a cluster

    Attributes:
        receive_queue (Queue): the queue we receive messages from
        send_queue (Queue): the queue we send messages through

        num_frame_workers (int): the number of frame workers we are allowed
        anim (MPAnimation): the animation
        frame_workers (list[FrameWorkerConnection]): the spawned frame workers

        worker_id (str): the id for this worker
        dpi (int or float): the number of pixels per inch
        frame_size (tuple[float, float]): the number of inches in width / height
        fps (int): the number of frames per second
        ffmpeg_logfile (str): the path to the logfile for ffmpeg
        outfile (str): the path to the file we are saving the video
        batch_size (int): how many points are being sent through the network
        layer_name (str): the name of the layer we are handling
        layer_size (int): the number of nodes in our layer
        output_dim (int): the number of labels / the output embedded dim
        sample_labels_file (str): the file that contains the sample labels
        hid_acts_file (str): the file that contains the hidden activations

        match_mean_comps_file (str): the file that we are storing the match-means info in
        match_mean_comps (np.memmap): the memmap'd means_comp
        match_mean_comps_torch (torch.tensor): torch tensor backed by match_means_comp

        hidacts (np.memmap): the memory mapped hid_acts file
        hidacts_torch (torch.tensor): the tensor backed with hidacts

        sample_labels (np.memmap): the memory mapped sample labels file
        sample_labels_torch (torch.tensor): the tensor backed with sample_labaels
    """

    def __init__(self, receive_queue: Queue, send_queue: Queue):
        self.receive_queue = receive_queue
        self.send_queue = send_queue

        self.num_frame_workers = None
        self.anim = None
        self.frame_workers = None

        self.worker_id = None
        self.dpi = None
        self.frame_size = None
        self.fps = None
        self.ffmpeg_logfile = None
        self.outfile = None
        self.batch_size = None
        self.layer_name = None
        self.layer_size = None
        self.output_dim = None
        self.sample_labels_file = None
        self.hid_acts_file = None

        self.match_mean_comps_file = None
        self.match_mean_comps = None
        self.match_mean_comps_torch = None

        self.hidacts = None
        self.hidacts_torch = None
        self.sample_labels = None
        self.sample_labels_torch = None

    def _read_start(self):
        """Reads the start message from the receive queue
        """

        msg = self.receive_queue.get(timeout=15)
        if msg[0] != 'start':
            raise RuntimeError(f'expected start message got {msg} ({msg[0]} != \'start\')')

        self.worker_id = str(msg[1]['worker_id'])
        self.dpi = msg[1]['dpi']
        self.frame_size = msg[1]['frame_size']
        self.fps = msg[1]['fps']
        self.ffmpeg_logfile = msg[1]['ffmpeg_logfile']
        self.num_frame_workers = msg[1]['num_frame_workers']
        self.outfile = msg[1]['outfile']
        self.batch_size = msg[1]['batch_size']
        self.layer_name = msg[1]['layer_name']
        self.layer_size = msg[1]['layer_size']
        self.output_dim = msg[1]['output_dim']
        self.sample_labels_file = msg[1]['sample_labels_file']
        self.hid_acts_file = msg[1]['hid_acts_file']

        if not isinstance(self.frame_size, tuple):
            raise ValueError(f'expected frame_size is tuple, got {self.frame_size} (msg={msg})')
        if len(self.frame_size) != 2:
            raise ValueError(f'expected frame_size has len 2, got {len(self.frame_size)}')

    def _prepare_mmaps(self):
        output_folder = os.path.dirname(self.outfile)
        self.match_mean_comps_file = os.path.join(output_folder, f'mean_comps_{self.worker_id}.bin')
        self.match_mean_comps = np.memmap(
            self.match_mean_comps_file, dtype='uint8', mode='w+',
            shape=(pca_ff.PCTrajectoryFFSnapshotMatchInfo.get_expected_len(3, self.output_dim)))
        self.match_mean_comps_torch = torch.from_numpy(self.match_mean_comps)

        self.hidacts = np.memmap(self.hid_acts_file, dtype='float64',
                                     mode='r', shape=(self.batch_size, self.layer_size))
        self.hidacts_torch = torch.from_numpy(self.hidacts)

        self.sample_labels = np.memmap(self.sample_labels_file, dtype='int32',
                                       mode='r', shape=(self.batch_size,))
        self.sample_labels_torch = torch.from_numpy(self.sample_labels)

    def _update_match(self):
        print(f'layer worker get_hidden_pcs; self.hidacts_torch.shape={self.hidacts_torch.shape}')
        pc_vals, pc_vecs = pca.get_hidden_pcs(self.hidacts_torch, 3)
        print('layer worker project_to_pcs')
        projected = pca.project_to_pcs(self.hidacts_torch, pc_vecs, out=None)
        print(f'layer worker create snap (pc_vals.shape={pc_vals.shape})')
        snap = pca_ff.PCTrajectoryFFSnapshot(pc_vecs, pc_vals, projected, self.sample_labels_torch)
        print('layer worker create match')
        match_info = pca_ff.PCTrajectoryFFSnapshotMatchInfo.create(snap)
        print(f'layer worker mmap mean_comps; trying to map shape {match_info.mean_comps.shape} dtype {match_info.mean_comps.dtype} into {self.match_mean_comps_torch.shape} dtype {self.match_mean_comps_torch.dtype}')
        self.match_mean_comps_torch[:] = match_info.mean_comps
        print('layer worker _update_match finished successfully')

    def _close_mmaps(self):
        self.match_mean_comps_torch = None
        self.match_mean_comps._mmap.close() # pylint: disable=protected-access
        self.match_mean_comps = None
        os.remove(self.match_mean_comps_file)
        self.match_mean_comps_file = None

    def _spawn_workers(self):
        """Initializes the animation and spawns the frame workers
        """

        self.anim = MPAnimation(
            self.dpi, (int(self.frame_size[0] * self.dpi), int(self.frame_size[1] * self.dpi)),
            self.fps, self.outfile, self.ffmpeg_logfile)

        self.anim.start()

        self.frame_workers = []
        for _ in range(self.num_frame_workers):
            jobq = Queue()
            imgq = Queue()
            ackq = Queue()
            proc = Process(target=_frame_worker_target,
                           args=(jobq, imgq, ackq, 'asap', self.hid_acts_file,
                                 self.sample_labels_file, self.match_mean_comps_file,
                                 self.batch_size, self.layer_size, self.output_dim,
                                 self.frame_size[0], self.frame_size[1], self.dpi))
            self.anim.register_queue(imgq)
            conn = FrameWorkerConnection(proc, jobq, ackq)
            self.frame_workers.append(conn)

            proc.start()

    def _dispatch_frame(self, rotation: float, title: str, index: int):
        start = time.time()
        while True:
            for worker in self.frame_workers:
                worker.check_ack()
                if not worker.awaiting_ack:
                    worker.send_job(rotation, title, index)
                    return

            if time.time() - start > 15000:
                raise RuntimeError(f'timeout occurred while trying to dispatch frame')

            self.anim.do_work()
            time.sleep(0)

    def _wait_all_acks(self):
        start = time.time()
        while True:
            waiting_ack = False
            for worker in self.frame_workers:
                worker.check_ack()
                waiting_ack = waiting_ack or worker.awaiting_ack

            if not waiting_ack:
                break

            if time.time() - start > 15000:
                raise RuntimeError(f'timeout while waiting for frame workers to acknowledge frame')

            self.anim.do_work()
            time.sleep(0)

    def _shutdown_all(self):
        for worker in self.frame_workers:
            worker.send_end()

        while True:
            waiting_end = False
            for worker in self.frame_workers:
                if not worker.check_end():
                    waiting_end = True

            if not waiting_end:
                break
            self.anim.do_work()
            time.sleep(0)

        while self.anim.process_frame():
            pass
        self.anim.finish()

    def work(self):
        """Handles the work required to render this layer"""
        self._read_start()
        self._prepare_mmaps()
        self._spawn_workers()

        print('layer worker started')
        rot_time = 0
        frame_counter = 0
        work_counter = 0
        last_work_time = time.time()
        while True:
            if not self.receive_queue.empty():
                msg = self.receive_queue.get_nowait()
                last_work_time = time.time()
            else:
                if time.time() - last_work_time > 15000:
                    raise RuntimeError(f'layer worker received no work to do and timed out')
                self.anim.do_work()
                time.sleep(0)
                continue
            if msg[0] == 'hidacts_done':
                break
            if msg[0] != 'hidacts':
                raise RuntimeError(f'unexpected msg: {msg} (expected hidacts or hidacts_done)')
            print('layer worker received work')
            if work_counter % 100 == 0:
                print('layer worker updating matching')
                self._update_match()
                print('layer worker updated match')
            work_counter += 1
            epoch = msg[1]
            for _ in range(FRAMES_PER_TRAIN):
                rot_prog = ROTATION_EASING(rot_time / MS_PER_ROTATION)
                rot = 45 + 360 * rot_prog
                self._dispatch_frame(rot, f'{self.layer_name} (epoch {epoch})', frame_counter)
                frame_counter += 1
                rot_time = (rot_time + FRAME_TIME) % MS_PER_ROTATION
            self._wait_all_acks()
            print('layer worker acking')
            self.send_queue.put(('hidacts',))

        self._shutdown_all()
        self._close_mmaps()
        self.send_queue.put(('videos_done',))

        while not self.send_queue.empty():
            time.sleep(0)

def _worker_target(receive_queue, send_queue):
    worker = LayerWorker(receive_queue, send_queue)
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
            msg = self.receive_queue.get(timeout=15)
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
        msg = self.receive_queue.get(timeout=15)
        if msg[0] != 'videos_done':
            raise ValueError(f'expected videos_done acknowledge')
        while self.process.is_alive():
            time.sleep(0.01)
        print('layer worker closed gracefully')

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

        self.dpi = 100
        self.fps = FPS
        self.frame_size = FRAME_SIZE


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
        for lyr in range(1, len(self.layers)):
            send_queue = Queue()
            receive_queue = Queue()
            proc = Process(target=_worker_target, args=(send_queue, receive_queue)) # swapped
            # cannot set to daemonic - it has children!
            proc.start()

            send_queue.put((
                'start',
                {
                    'worker_id': str(lyr),
                    'outfile': os.path.join(self.output_folder, f'layer_{lyr}.mp4'),
                    'batch_size': self.batch_size,
                    'layer_name': self.layer_names[lyr],
                    'layer_size': layer_sizes[lyr],
                    'output_dim': context.test_pwl.output_dim,
                    'sample_labels_file': self.sample_labels_file,
                    'hid_acts_file': self.hid_acts_files[lyr],
                    'dpi': self.dpi,
                    'fps': self.fps,
                    'frame_size': self.frame_size,
                    'ffmpeg_logfile': os.path.join(self.output_folder, f'layer_{lyr}_ffmpeg.log'),
                    'num_frame_workers': 4
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

    def finished(self, context: GenericTrainingContext, result: dict):
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








