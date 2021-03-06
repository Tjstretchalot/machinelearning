"""This module produces an animated 3d pc-plot through training. It outputs 1 video
per layer.

The raw arrays are sent via memory sharing.
"""

import typing
import os
from multiprocessing import Queue, Process
import time
import io
import traceback
import datetime

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
from shared.perf_stats import LoggingPerfStats
from shared.myqueue import ZeroMQQueue

INDEPENDENT_SCALING = False
"""True to scale each axis independently, false for each axis to have the same scale"""

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

TIMEOUT_TIME = 300 # how long we wait before we give up on queues, should be long unless debugging
SYNC_WORK_ITEMS = 1000
SKIP_TRAINS = 0 # number of frames skipped between every hidacts
FRAMES_PER_TRAIN = 4
NUM_FRAME_WORKERS = 4
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
        receive_queue (ZeroMQQueue): the queue we receive messages from. Messages come in the form of
            either ('frame', rotation, title, index) or ('end',)
        img_queue (ZeroMQQueue): the queue we send messages in the form (index, rawdata) where rawdata
            is the bytes that ought to be sent to the ffmpeg process.
        ack_queue (ZeroMQQueue): the queue we send messages in the form ('ack',) once we have completed
            a frame. this lets the layer worker know its safe to tell the main thread that the
            hidden activations file can be modified
        ack_mode (str): determines when we send the 'ack' message through the ack_queue
                'asap': we send as soon as modifying hidden_acts would not hurt our rendering
                'ready': we send as soon as we're ready for a new hidden_acts
                'both': ack both times

        hacts_file (str): the path to the hidden activations file, which contains the memmappable
            [batch_size x layer_size] data
        labels_file (str): the path to the labels file which contains the [batch_size] data.
            we only need to load this temporarily for the purpose of blitting the colors and
            from there on we only need to rotate images, however we memmap in order to use
            the pca calculations more smoothly
        match_mean_comps_file (str): the path to the file which will contain the 'mean_comps' that
            we should match our snapshots to; updated occassionally via a 'match' call and ensures
            deterministic behavior among frame workers
        perf_file (str): the path to the file which we will log performance information to

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

        perf (perf_stats.PerfStats)

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



    def __init__(self, receive_queue: ZeroMQQueue, img_queue: ZeroMQQueue, ack_queue: ZeroMQQueue, ack_mode: str,
                 hacts_file: str, labels_file: str, match_mean_comps_file: str, batch_size: int,
                 layer_size: int, output_dim: int, w_in: float, h_in: float, dpi: int,
                 perf_file: str):
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

        self.perf = LoggingPerfStats(None, perf_file)

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
            c=self.snapshot.projected_sample_labels.numpy(),
            cmap=mpl.cm.get_cmap('Set1'))
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

        if INDEPENDENT_SCALING:
            self.axes.set_xlim(float(snapsh.projected_samples[:, 0].min()),
                            float(snapsh.projected_samples[:, 0].max()))
            self.axes.set_ylim(float(snapsh.projected_samples[:, 1].min()),
                            float(snapsh.projected_samples[:, 1].max()))
            self.axes.set_zlim(float(snapsh.projected_samples[:, 2].min()),
                            float(snapsh.projected_samples[:, 2].max()))
        else:
            minlim = float(snapsh.projected_samples.min())
            maxlim = float(snapsh.projected_samples.max())
            self.axes.set_xlim(minlim, maxlim)
            self.axes.set_ylim(minlim, maxlim)
            self.axes.set_zlim(minlim, maxlim)

        self.axes.view_init(30, self.rotation)
        self.axtitle.set_text(self.title)

    def _create_frame(self) -> bytes:
        hndl = io.BytesIO()
        self.figure.set_size_inches(self.w_in, self.h_in)
        self.figure.savefig(hndl, format=self.frame_format, dpi=self.dpi)
        return hndl.getvalue()

    def work(self):
        """Initializes and produces frames until the end message is received"""
        try:
            while self.do_work():
                pass
        except:
            self.perf.loghandle.flush()
            print('--CRASH--', file=self.perf.loghandle)
            traceback.print_exc(file=self.perf.loghandle)
            self.perf.loghandle.flush()
            raise

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
            msg = self.receive_queue.get() # cannot timeout here & support extra workers
            self.last_job = time.time()
            if msg[0] == 'end':
                self._close_mmaps()
                self._close_figure()
                self.state = 3
                return False
            self.rotation, self.title, self.index, job_sent_at = msg[1:]
            self.perf.enter('RECEIVE_JOB_DELAY', force_time=job_sent_at)
            self.perf.exit()
            self.state = 2
            return True
        if self.state == 2:
            self.perf.enter('GET_SNAPSHOT')
            self._get_snapshot()
            self.perf.exit()
            if self.ack_mode in ('asap', 'both'):
                self.perf.enter('ACK_ASAP')
                self.ack_queue.put(('ack', time.time()))
                self.perf.exit()
            if self.figure is None:
                self._init_figure()
            self.perf.enter('RENDER')
            self.perf.enter('SETUP_FRAME')
            self._setup_frame()
            self.perf.exit()
            self.perf.enter('CREATE_FRAME')
            frm = self._create_frame()
            self.perf.exit()
            self.perf.enter('IMG_QUEUE_PUT')
            self.img_queue.put((self.index, frm))
            self.perf.exit()
            self.perf.exit()
            if self.ack_mode in ('ready', 'both'):
                self.perf.enter('ACK_READY')
                self.ack_queue.put(('ack', time.time()))
                self.perf.exit()
            self.state = 1
            return True

        return False

def _frame_worker_target(receive_queue: ZeroMQQueue, img_queue: ZeroMQQueue, ack_queue: ZeroMQQueue, ack_mode: str,
                         hacts_file: str, labels_file: str, match_means_comp_file: str,
                         batch_size: int, layer_size: int, output_dim: int, w_in: float,
                         h_in: float, dpi: int, perf_file: str):
    receive_queue = ZeroMQQueue.deser(receive_queue)
    img_queue = ZeroMQQueue.deser(img_queue)
    ack_queue = ZeroMQQueue.deser(ack_queue)
    worker = FrameWorker(receive_queue, img_queue, ack_queue, ack_mode, hacts_file, labels_file,
                         match_means_comp_file, batch_size, layer_size, output_dim, w_in, h_in, dpi,
                         perf_file)
    worker.work()

class FrameWorkerConnection:
    """Describes the connection a LayerWorker has with a FrameWorker

    Attributes:
        process (Process): the process that the frame worker is running on
        job_queue (ZeroMQQueue): the queue which we send jobs to
        ack_queue (ZeroMQQueue): the queue which we receive acks from

        awaiting_asap_ack (bool): True if we haven't seen an ack from the last job yet
        awaiting_ready_ack (bool): True if we haven't seen a ready ack for the last job

        ack_mode (str): one of 'asap', 'ready', 'both'
    """

    def __init__(self, process: Process, job_queue: ZeroMQQueue, ack_queue: ZeroMQQueue,
                 ack_mode: str):
        self.process = process
        self.job_queue = job_queue
        self.ack_queue = ack_queue
        self.ack_mode = ack_mode

        self.awaiting_asap_ack = False
        self.awaiting_ready_ack = False

    def check_ack(self):
        """Checks if we have any acks waiting in the queue. Returns the ack
        sent at time if we receive one
        """
        if not self.awaiting_ready_ack:
            return None
        if not self.ack_queue.empty():
            ack = self.ack_queue.get_nowait()
            if ack is None:
                return None
            if self.awaiting_asap_ack:
                self.awaiting_asap_ack = False
            else:
                self.awaiting_ready_ack = False

            return ack[1]

    def wait_ack(self, ready=True):
        """Waits for the acknowledgement of the job from the frame worker"""
        if not self.awaiting_ready_ack:
            return
        if not ready and not self.awaiting_asap_ack:
            return
        self.ack_queue.get(timeout=TIMEOUT_TIME)
        if self.awaiting_asap_ack:
            self.awaiting_asap_ack = False
        else:
            self.awaiting_ready_ack = False

    def send_job(self, rotation: float, title: str, index: int):
        """Tells the frame worker to produce an image of the given rotation and title. The
        index is the index of the frame within the video, where 0 is the first frame.

        Arguments:
            rotation (float): in degrees around the y axis
            title (str): the title of the image
            index (int): the index in the video of the frame
        """
        self.wait_ack()
        self.job_queue.put(('frame', rotation, title, index, time.time()))
        self.awaiting_asap_ack = self.ack_mode in ('asap', 'both')
        self.awaiting_ready_ack = self.ack_mode in ('ready', 'both')

    def send_end(self):
        """Tells the worker to shutdown"""
        self.job_queue.put(('end',))

    def wait_end(self):
        """Waits for the worker to shutdown"""
        self.process.join()

    def check_end(self) -> bool:
        """Checks if the worker has shutdown yet"""
        return not self.process.is_alive()

class LayerEncoderWorker:
    """Describes a worker which simply manages sending the image queue through to the
    ffmpeg instance. This was originally part of the LayerWorker but is actually fairly
    expensive

    Attributes:
        receive_queue (ZeroMQQueue): the queue we receive meta messages (i.e. shutdown) from
        response_queue (ZeroMQQueue): the queue we send meta messages through
        img_queue (ZeroMQQueue): the queue that images are passed in through

        dpi (int): pixels per inch
        frame_size (tuple): the width/height in inches to render at
        fps (int): number of frames per second
        outfile (str): the file to save at
        ffmpeg_logfile (str): where we can save ffmpeg performance information
        logfile (str): where we can save our debug information
        loghandle (file): the file handle to logfile

        anim (MPAnimation): the animation
    """

    def __init__(self, receive_queue: ZeroMQQueue, response_queue: ZeroMQQueue, dpi: int,
                 frame_size: tuple, fps: int, outfile: str, ffmpeg_logfile: str):
        self.receive_queue = receive_queue
        self.response_queue = response_queue
        self.img_queue = None
        self.dpi = dpi
        self.frame_size = frame_size
        self.fps = fps
        self.outfile = outfile
        self.ffmpeg_logfile = ffmpeg_logfile

        self.logfile = os.path.join(os.path.dirname(outfile),
                                    os.path.splitext(os.path.basename(outfile))[0] + '_encoder.log')
        self.loghandle = open(self.logfile, 'w')

        self.anim = None

    def prepare(self):
        """Prepares this worker to do work"""
        self.img_queue = ZeroMQQueue.create_recieve()
        self.response_queue.put(self.img_queue.port)

        self.anim = MPAnimation(
            self.dpi, (int(self.frame_size[0] * self.dpi), int(self.frame_size[1] * self.dpi)),
            self.fps, self.outfile, self.ffmpeg_logfile)
        self.anim.register_queue(self.img_queue)
        self.anim.start()

    def do_work(self):
        """Tries to do some work and returns True if we did something False if we did nothing"""
        return self.anim.do_work()

    def shutdown(self):
        """Prepares this worker to be shutdown"""
        while self.do_work():
            pass
        self.anim.finish()
        self.anim = None

    def work(self):
        """Should be called after initialization to work until we receive a shutdown message"""

        try:
            print('Preparing...', file=self.loghandle)
            self.prepare()

            print('Working..', file=self.loghandle)
            work_count = 0
            self.loghandle.flush()

            syncing = False
            while True:
                for _ in range(100):
                    if not self.do_work():
                        if syncing:
                            self.response_queue.put(('sync',))
                            syncing = False
                            self.loghandle.flush()
                        break
                    else:
                        work_count += 1
                        if work_count % 100 == 0:
                            print(f'{datetime.datetime.now()} - Finished work item {work_count}', file=self.loghandle)
                            if work_count % 500 == 0:
                                self.loghandle.flush()

                if not syncing and not self.receive_queue.empty():
                    msg = self.receive_queue.get()

                    if msg[0] == 'sync':
                        if syncing:
                            raise RuntimeError(f'receiving sync message while still syncing!')
                        syncing = True
                    elif msg[0] == 'shutdown':
                        break

                time.sleep(0.001)

            print(f'{datetime.datetime.now()} Received shutdown message', file=self.loghandle)
            self.loghandle.flush()

            print('Shutting down...', file=self.loghandle)
            self.loghandle.flush()
            self.shutdown()

            print('Successfully shutdown', file=self.loghandle)
            self.loghandle.close()
        except:
            traceback.print_exc(file=self.loghandle)
            self.loghandle.close()
            self.loghandle = None
            raise

def _layer_encoder_target(recq, respq, dpi, frame_size, fps, outfile, ffmpeg_logfile):
    recq = ZeroMQQueue.deser(recq)
    respq = ZeroMQQueue.deser(respq)
    worker = LayerEncoderWorker(recq, respq, dpi, frame_size, fps, outfile, ffmpeg_logfile)
    worker.work()

class LayerEncoderWorkerConnection:
    """Describes a connecter from the layerworker to the layerencoder worker

    Attributes:
        process (Process): the encoder worker process
        notif_queue (ZeroMQQueue): the put queue we can send notifications with
        ack_queue (ZeroMQQueue): the get queue we can receive messages from
    """

    def __init__(self, process: Process, notif_queue: ZeroMQQueue, ack_queue: ZeroMQQueue):
        self.process = process
        self.notif_queue = notif_queue
        self.ack_queue = ack_queue

    def sync(self):
        """Waits for the encoder worker to catch up"""
        start = time.time()
        self.notif_queue.put(('sync',))
        self.ack_queue.get()
        sync_time = time.time() - start
        if sync_time > 1:
            print(f'Syncing layer encoder took {sync_time:.2f}s')

    def shutdown(self):
        """Cleanly shuts down the encoder"""
        if not self.process.is_alive():
            return

        self.notif_queue.put(('shutdown',))
        while self.process.is_alive():
            time.sleep(0.001)

class LayerWorker:
    """Describes a worker instance. A worker manages rendering of a single layer. It itself
    will delegate to FrameWorkers which will render the individual frames, and then to a
    LayerEncoderWorker to push the animations to ffmpeg for encoding, which is typically the
    expensive part of the process.

    Example training a 7-layer network: the main thread delegates to 7 layer thread which each
        delegate to 4 frame threads and 7 encoder threads. A total of 37 threads are used for
        generating this plot, which is probably only going to work on a cluster

    Attributes:
        receive_queue (ZeroMQQueue): the queue we receive messages from
        send_queue (ZeroMQQueue): the queue we send messages through

        num_frame_workers (int): the number of frame workers we are allowed
        encoder (LayerEncoderWorkerConnection): the encoder connection
        frame_workers (list[FrameWorkerConnection]): the spawned frame workers

        worker_id (str): the id for this worker
        dpi (int or float): the number of pixels per inch
        frame_size (tuple[float, float]): the number of inches in width / height
        fps (int): the number of frames per second
        ffmpeg_logfile (str): the path to the logfile for ffmpeg
        perf_logfile (str): the path we are saving performance info
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
        match_info (PCATrajectoryFFSnapshotMatchInfo): the match info the above correspond to

        hidacts (np.memmap): the memory mapped hid_acts file
        hidacts_torch (torch.tensor): the tensor backed with hidacts

        sample_labels (np.memmap): the memory mapped sample labels file
        sample_labels_torch (torch.tensor): the tensor backed with sample_labaels

        perf (LoggingPerfStats): the layer worker performance tracker
    """

    def __init__(self, receive_queue: ZeroMQQueue, send_queue: ZeroMQQueue):
        self.receive_queue = receive_queue
        self.send_queue = send_queue

        self.num_frame_workers = None
        self.encoder = None
        self.frame_workers = None

        self.worker_id = None
        self.dpi = None
        self.frame_size = None
        self.fps = None
        self.ffmpeg_logfile = None
        self.perf_logfile = None
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
        self.match_info = None

        self.hidacts = None
        self.hidacts_torch = None
        self.sample_labels = None
        self.sample_labels_torch = None

        self.perf = None

    def _read_start(self):
        """Reads the start message from the receive queue
        """

        msg = self.receive_queue.get(timeout=TIMEOUT_TIME)
        if msg[0] != 'start':
            raise RuntimeError(f'expected start message got {msg} ({msg[0]} != \'start\')')

        self.worker_id = str(msg[1]['worker_id'])
        self.dpi = msg[1]['dpi']
        self.frame_size = msg[1]['frame_size']
        self.fps = msg[1]['fps']
        self.ffmpeg_logfile = msg[1]['ffmpeg_logfile']
        self.perf_logfile = msg[1]['perf_logfile']
        self.num_frame_workers = msg[1]['num_frame_workers']
        self.outfile = msg[1]['outfile']
        self.batch_size = msg[1]['batch_size']
        self.layer_name = msg[1]['layer_name']
        self.layer_size = msg[1]['layer_size']
        self.output_dim = msg[1]['output_dim']
        self.sample_labels_file = msg[1]['sample_labels_file']
        self.hid_acts_file = msg[1]['hid_acts_file']

        if isinstance(self.frame_size, list):
            self.frame_size = tuple(self.frame_size)
        elif not isinstance(self.frame_size, tuple):
            raise ValueError(f'expected frame_size is tuple, got {self.frame_size} (msg={msg})')
        if len(self.frame_size) != 2:
            raise ValueError(f'expected frame_size has len 2, got {len(self.frame_size)}')

        self.perf = LoggingPerfStats(self.worker_id, self.perf_logfile)

    def _prepare_mmaps(self):
        output_folder = os.path.dirname(self.outfile)
        self.match_mean_comps_file = os.path.join(output_folder, f'mean_comps_{self.worker_id}.bin')
        self.match_mean_comps = np.memmap(
            self.match_mean_comps_file, dtype='uint8', mode='w+',
            shape=(pca_ff.PCTrajectoryFFSnapshotMatchInfo.get_expected_len(3, self.output_dim)))
        self.match_mean_comps_torch = torch.from_numpy(self.match_mean_comps)
        self.match_info = pca_ff.PCTrajectoryFFSnapshotMatchInfo(
            3, self.layer_size, self.output_dim, self.match_mean_comps_torch)

        self.hidacts = np.memmap(self.hid_acts_file, dtype='float64',
                                     mode='r', shape=(self.batch_size, self.layer_size))
        self.hidacts_torch = torch.from_numpy(self.hidacts)

        self.sample_labels = np.memmap(self.sample_labels_file, dtype='int32',
                                       mode='r', shape=(self.batch_size,))
        self.sample_labels_torch = torch.from_numpy(self.sample_labels)

    def _update_match(self):
        pc_vals, pc_vecs = pca.get_hidden_pcs(self.hidacts_torch, 3)
        projected = pca.project_to_pcs(self.hidacts_torch, pc_vecs, out=None)
        snap = pca_ff.PCTrajectoryFFSnapshot(pc_vecs, pc_vals, projected, self.sample_labels_torch)
        self.match_info.match(snap)
        match_info = pca_ff.PCTrajectoryFFSnapshotMatchInfo.create(snap)
        self.match_mean_comps_torch[:] = match_info.mean_comps

    def _close_mmaps(self):
        self.match_mean_comps_torch = None
        self.match_mean_comps._mmap.close() # pylint: disable=protected-access
        self.match_mean_comps = None
        self.match_info = None
        os.remove(self.match_mean_comps_file)
        self.match_mean_comps_file = None

    def _spawn_workers(self):
        """Initializes the animation and spawns the frame workers
        """

        enc_notif_queue = ZeroMQQueue.create_send()
        enc_notif_squeue = ZeroMQQueue.create_recieve()
        proc = Process(target=_layer_encoder_target,
                       args=(enc_notif_queue.serd(), enc_notif_squeue.serd(),
                             self.dpi, self.frame_size, self.fps, self.outfile,
                             self.ffmpeg_logfile))
        proc.start()

        imgq_port = enc_notif_squeue.get()
        imgq = ZeroMQQueue.create_recieve(port=imgq_port)

        self.encoder = LayerEncoderWorkerConnection(proc, enc_notif_queue, enc_notif_squeue)

        self.frame_workers = []
        closure = lambda x: x
        for idx in range(self.num_frame_workers):
            jobq = ZeroMQQueue.create_send()
            ackq = ZeroMQQueue.create_recieve()
            ackm = 'both'
            proc = Process(target=_frame_worker_target,
                           args=(jobq.serd(), imgq.serd(), ackq.serd(), ackm, self.hid_acts_file,
                                 self.sample_labels_file, self.match_mean_comps_file,
                                 self.batch_size, self.layer_size, self.output_dim,
                                 self.frame_size[0], self.frame_size[1], self.dpi,
                                 os.path.join(
                                     os.path.dirname(self.outfile),
                                     # do not change below to just idx because it wont be closed
                                     f'layer_{self.worker_id}_frame_{closure(idx)}.log')))
            conn = FrameWorkerConnection(proc, jobq, ackq, ackm)
            self.frame_workers.append(conn)

            proc.start()


    def _dispatch_frame(self, rotation: float, title: str, index: int):
        start = time.time()
        while True:
            for worker in self.frame_workers:
                worker.check_ack()
                if not worker.awaiting_ready_ack:
                    worker.send_job(rotation, title, index)
                    return

            if time.time() - start > TIMEOUT_TIME:
                raise RuntimeError(f'timeout occurred while trying to dispatch frame')

    def _wait_all_acks(self):
        """Waits for the ASAP acks"""
        start = time.time()
        while True:
            waiting_ack = False
            for worker in self.frame_workers:
                ack_start = worker.check_ack()
                waiting_ack = waiting_ack or worker.awaiting_asap_ack

                if ack_start is not None:
                    if worker.awaiting_ready_ack:
                        self.perf.enter('ASAP_ACK_DELAY', force_time=ack_start)
                    else:
                        self.perf.enter('READY_ACK_DELAY', force_time=ack_start)
                    self.perf.exit()

            if not waiting_ack:
                break

            if time.time() - start > TIMEOUT_TIME:
                raise RuntimeError(f'timeout while waiting for frame workers to acknowledge frame')

    def _shutdown_all(self):
        for worker in self.frame_workers:
            worker.send_end()
        start = time.time()
        printed_warn = False
        while True:
            waiting_end = False
            for worker in self.frame_workers:
                if not worker.check_end():
                    waiting_end = True

            if not waiting_end:
                break
            if (not printed_warn) and ((time.time() - start) > TIMEOUT_TIME):
                print(f'LayerWorker {self.worker_id} - frame workers taking a long time to close')
                printed_warn = True

        print(f'LayerWorker {self.worker_id} shutting down encoder')
        self.encoder.shutdown()
        print(f'LayerWorker {self.worker_id} successfully shutdown encoder')
        self.encoder = None

    def work(self):
        """Handles the work required to render this layer"""
        self._read_start()
        self._prepare_mmaps()
        self._spawn_workers()

        rot_time = 0
        frame_counter = 0
        work_counter = 0
        last_work_time = time.time()
        self.perf.enter('RECEIVE_QUEUE')
        while True:
            if not self.receive_queue.empty():
                msg = self.receive_queue.get_nowait()
                self.perf.exit()
                last_work_time = time.time()
            else:
                if time.time() - last_work_time > TIMEOUT_TIME:
                    raise RuntimeError(f'layer worker received no work to do and timed out')
                continue
            if msg[0] == 'hidacts_done':
                break
            if msg[0] != 'hidacts':
                raise RuntimeError(f'unexpected msg: {msg} (expected hidacts or hidacts_done)')
            if work_counter % 100 == 0:
                self.perf.enter('UPDATE_MATCH')
                self._update_match()
                self.perf.exit()
            if work_counter % SYNC_WORK_ITEMS == 0:
                self.perf.enter('SYNC_ENCODER')
                self.encoder.sync()
                self.perf.exit()

            work_counter += 1
            epoch = msg[1]
            for _ in range(FRAMES_PER_TRAIN):
                rot_prog = ROTATION_EASING(rot_time / MS_PER_ROTATION)
                rot = 45 + 360 * rot_prog
                self.perf.enter('DISPATCH_FRAME')
                self._dispatch_frame(rot, f'{self.layer_name} (epoch {epoch})', frame_counter)
                self.perf.exit()
                frame_counter += 1
                rot_time = (rot_time + FRAME_TIME) % MS_PER_ROTATION
            self.perf.enter('WAIT_ALL_ACKS')
            self._wait_all_acks()
            self.perf.exit()
            self.send_queue.put(('hidacts',))
            self.perf.enter('RECEIVE_QUEUE')

        print(f'LayerWorker {self.worker_id} shutting down')
        self.perf.enter('SHUTDOWN_ALL')
        self._shutdown_all()
        self.perf.exit()
        self.perf.close()
        self._close_mmaps()
        self.send_queue.put(('videos_done',))

def _worker_target(receive_queue, send_queue):
    worker = LayerWorker(ZeroMQQueue.deser(receive_queue), ZeroMQQueue.deser(send_queue))
    try:
        worker.work()
    except:
        traceback.print_exc()
        raise

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
            msg = self.receive_queue.get(timeout=TIMEOUT_TIME)
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
        msg = self.receive_queue.get(timeout=120)
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
        layer_indices (list[int]): the indicies in layer_names that we actually send to layer
            workers

        connections (list[WorkerConnection]): the connections to the workers
        skip_counter (int): number of hidacts skipped since last sent one

        batch_size (int): the number of points we are plotting
        sample_labels [np.ndarray]: the memmap'd int32 array we share labels with
        sample_points [np.ndarray]: the unmapped float64 array we store the points in

        layers [list[np.ndarray]]: the list of memmap'd float64 arrays we share hid_acts with

        sample_labels_torch [torch.tensor]: torch.from_numpy(sample_labels)
        sample_points_torch [torch.tensor]: torch.from_numpy(sample_points)

        sample_labels_file (str): the path to the sample labels mmap'd file
        hid_acts_files (list[str]): the paths to the hidden activation files (by layer) mmap'd
    """

    def __init__(self, output_path: str, layer_names: typing.List[str], exist_ok: bool = False,
                 layer_indices=None):
        """
        Args:
            output_path (str): either the output folder or the output archive
            layer_names (str): a list of layer names starting with 'input' and
                ending with 'output'
            exist_ok (bool, optional): Defaults to False. If True, if the output
                archive already exists it will be deleted. Otherwise, if the output
                archive already exists an error will be raised
            layer_indices (list[int], optional): Default to None. If specified, only the
                given layers are rendered (where 0 is the input and -1 is output). Otherwise
                defaults to [1:]
        """

        _, self.output_folder = mutils.process_outfile(output_path, exist_ok)
        self.layer_names = layer_names
        self.layer_indices = layer_indices

        self.connections = None
        self.skip_counter = None

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

        if self.layer_indices is None:
            self.layer_indices = [i for i in range(1, len(acts.hid_acts))]
        layer_sizes = []
        self.hid_acts_files = []
        self.layers = []
        for idx in self.layer_indices:
            lyr = acts.hid_acts[idx]
            filepath = os.path.join(self.output_folder, f'hid_acts_{idx}.bin')
            self.hid_acts_files.append(filepath)
            layer_sizes.append(int(lyr.shape[1]))
            self.layers.append(np.memmap(filepath, dtype='float64', mode='w+', shape=(self.batch_size, int(lyr.shape[1]))))

        self.skip_counter = 0
        self.connections = []
        for lyrsidx, lyr in enumerate(self.layer_indices):
            send_queue = ZeroMQQueue.create_send()
            receive_queue = ZeroMQQueue.create_recieve()
            proc = Process(target=_worker_target, args=(send_queue.serd(), receive_queue.serd())) # swapped
            # cannot set to daemonic - it has children!
            proc.start()

            send_queue.put((
                'start',
                {
                    'worker_id': str(lyrsidx),
                    'outfile': os.path.join(self.output_folder, f'layer_{lyr}.mp4'),
                    'batch_size': self.batch_size,
                    'layer_name': self.layer_names[lyr],
                    'layer_size': layer_sizes[lyrsidx],
                    'output_dim': context.test_pwl.output_dim,
                    'sample_labels_file': self.sample_labels_file,
                    'hid_acts_file': self.hid_acts_files[lyrsidx],
                    'dpi': self.dpi,
                    'fps': self.fps,
                    'frame_size': self.frame_size,
                    'ffmpeg_logfile': os.path.join(self.output_folder, f'layer_{lyr}_ffmpeg.log'),
                    'perf_logfile': os.path.join(self.output_folder, f'layer_{lyr}_perf.log'),
                    'num_frame_workers': NUM_FRAME_WORKERS
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
        for lyrs_idx, idx in enumerate(self.layer_indices):
            self.layers[lyrs_idx][:] = nhacts.hid_acts[idx]
        for connection in self.connections:
            connection.send_hidacts(int(context.shared['epochs'].epochs))

    def post_train(self, context: GenericTrainingContext, loss: float):
        """Feeds hidden activations to the network"""
        if self.skip_counter >= SKIP_TRAINS:
            self._send_hidacts(context)
            self.skip_counter = 0
        else:
            self.skip_counter += 1

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

        self.sample_labels._mmap.close() # pylint: disable=protected-access
        self.sample_labels = None

        self.sample_points = None

        for lyr in self.layers:
            lyr._mmap.close() # pylint: disable=protected-access

        self.layers = None

        os.remove(self.sample_labels_file)

        for hafile in self.hid_acts_files:
            os.remove(hafile)

        self.sample_labels_file = None
        self.hid_acts_files = None

        zipdir(self.output_folder)








