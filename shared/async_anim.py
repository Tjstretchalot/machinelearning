"""Provides tools to animate figures in push frame / pop frame style
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import writers, subprocess_creation_flags

import multiprocessing as mp
import subprocess as sp
import logging
import typing
import os


_log = logging.getLogger(__name__)


class AsyncAnimation:
    """Describes an animation which needs to be explicitly told when a frame is ready.
    This does not support interactive mode. This does not support other animations.
    Only one such animation can occur per process in a given instant. See MPAsyncAnimation
    for a multi-processed wrapper of this class.

    Attributes:
        fig (matplotlib.figure.Figure): the figure object that is used to draw, resize,
            etc.

        _my_rc_ctx (typing.Any): the result from mpl.rc_context() that must be exitted
            when saving is complete

        _save_ctx (typing.Any): the result from writer.saving that must be exitted when
            saving is complete

        _writer (typing.Any): the writer we are using

        _savefig_kwargs (dict): args passed to writer grab_frame
    """

    def __init__(self, fig: mpl.figure.Figure):
        if not isinstance(fig, mpl.figure.Figure):
            raise ValueError(f'expected fig is mpl.figure.Figure, got {fig} (type={type(fig)})')
        self.fig = fig

        self._my_rc_ctx = None
        self._save_ctx = None
        self._writer = None
        self._savefig_kwargs = None

    def prepare_save(self, filename, writer=None, fps=5, dpi=None, codec=None,
             bitrate=None, extra_args=None, metadata=None, extra_anim=None,
             savefig_kwargs=None):
        """See https://matplotlib.org/_modules/matplotlib/animation.html#Animation save, except
        with fps not None for default"""

        if writer is None:
            writer = mpl.rcParams['animation.writer']
        elif (not isinstance(writer, str) and
              any(arg is not None
                  for arg in (fps, codec, bitrate, extra_args, metadata))):
            raise RuntimeError('Passing in values for arguments '
                               'fps, codec, bitrate, extra_args, or metadata '
                               'is not supported when writer is an existing '
                               'MovieWriter instance. These should instead be '
                               'passed as arguments when creating the '
                               'MovieWriter instance.')

        if savefig_kwargs is None:
            savefig_kwargs = {}

        if dpi is None:
            dpi = mpl.rcParams['savefig.dpi']
        if dpi == 'figure':
            dpi = self.fig.dpi

        if codec is None:
            codec = mpl.rcParams['animation.codec']

        if bitrate is None:
            bitrate = mpl.rcParams['animation.bitrate']


        # If we have the name of a writer, instantiate an instance of the
        # registered class.
        if isinstance(writer, str):
            if writer in writers.avail:
                writer = writers[writer](fps, codec, bitrate,
                                         extra_args=extra_args,
                                         metadata=metadata)
            else:
                _log.warning("MovieWriter {} unavailable. Trying to use {} "
                             "instead.".format(writer, writers.list()[0]))

                try:
                    writer = writers[writers.list()[0]](fps, codec, bitrate,
                                                        extra_args=extra_args,
                                                        metadata=metadata)
                except IndexError:
                    raise ValueError("Cannot save animation: no writers are "
                                     "available. Please install ffmpeg to "
                                     "save animations.")
        _log.info('AsyncAnimation.save using %s', type(writer))

        if 'bbox_inches' in savefig_kwargs:
            _log.warning("Warning: discarding the 'bbox_inches' argument in "
                         "'savefig_kwargs' as it may cause frame size "
                         "to vary, which is inappropriate for animation.")
            savefig_kwargs.pop('bbox_inches')

        self._my_rc_ctx = mpl.rc_context()
        self._my_rc_ctx.__enter__()

        if mpl.rcParams['savefig.bbox'] == 'tight':
            _log.info("Disabling savefig.bbox = 'tight', as it may cause "
                        "frame size to vary, which is inappropriate for "
                        "animation.")
            mpl.rcParams['savefig.bbox'] = None

        self._save_ctx = writer.saving(self.fig, filename, dpi)
        self._save_ctx.__enter__()

        self._writer = writer
        self._savefig_kwargs = savefig_kwargs

    def on_frame(self, dirty_arts):
        """Should be invoked with the tuple of dirty artists, though they
        are currently unused"""
        # pre_draw -> if blitting, blit clear
        # draw -> if blitting, set dirty artists to animated?
        # post_draw -> blit draw or draw idle
        self.fig.canvas.draw_idle()
        self._writer.grab_frame(**self._savefig_kwargs)

    def on_finish(self):
        """Should be called after all frames have been written"""
        self._save_ctx.__exit__(None, None, None)
        self._save_ctx = None
        self._my_rc_ctx.__exit__(None, None, None)
        self._my_rc_ctx = None
        self._writer = None
        self._savefig_kwargs = None

class MPAnimation:
    """Describes a multiprocessing-friendly animation. This is a more refined
    version of the async animation which incorporates the writer directly.

    The idea behind this animation is it becomes possible to queue frames from
    other processes. Frames are sent to this animation via a queue which then
    reorders them and writes them to the ffmpeg writer as appropriate. The
    queue is sent the frame number which it uses to reorder. An error is raised
    if too many frames are waiting for a single frame.

    This may have as many input queues as required. Messages should be in the form
    (int, bytes) where int is the frame and bytes is the result from savefig stored
    into a BytesIO.

    Attributes:
        dpi (int or float): the number of pixels per inch of the figure
        fps (int): the number of frames per second
        outfile (str): where we are saving to
        logfile (str): where we dump ffmpeg output to

        frame_size (typing.Tuple[int, int]): the size of the frame in pixels,
            initialized right before the ffmpeg process is opened

        ffmpeg_proc (mp.Process): the process used to communicate with ffmpeg
        logfh (filehandler): the file handler used for logging the ffmpeg output

        next_frame (int): the index for the next frame to write to file
        ooo_frames (dict[int, bytes]): frames that arrived out of order. the key is the
            frame and the value are the bytes array from savefig

        receive_queues (list[mp.Queue]): the queues we are receiving frames from
    """

    def __init__(self, dpi: typing.Union[int, float], frame_size: typing.Tuple[int, int],
                 fps: int, outfile: str, logfile: str):
        if not isinstance(dpi, (int, float)):
            raise ValueError(f'expected dpi is number, got {dpi} (type={type(dpi)})')
        if not isinstance(frame_size, tuple):
            raise ValueError(f'expected frame_size is (width inches, height_inches) but got {frame_size} (type={type(frame_size)} != tuple)')
        if len(frame_size) != 2:
            raise ValueError(f'expected frame_size is (width inches, height inches) but len(frame_size)={len(frame_size)}')
        if not isinstance(frame_size[0], (int, float)) or not isinstance(frame_size[1], (int, float)):
            raise ValueError(f'expected frame_size is (width inches, height inches) but not both numbers (got {frame_size})')
        frame_size = (int(frame_size[0]), int(frame_size[1]))
        if not isinstance(fps, int):
            raise ValueError(f'expected fps is int, got {fps} (type={type(fps)})')
        if not isinstance(outfile, str):
            raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')

        wo_ext, ext = os.path.splitext(outfile)
        if ext == '':
            outfile = wo_ext + '.mp4'
        elif ext != '.mp4':
            raise ValueError(f'expected outfile is mp4 file, got {outfile} (ext={ext})')

        wo_ext, ext = os.path.splitext(logfile)
        if ext == '':
            logfile = wo_ext + '.txt'

        if os.path.exists(outfile):
            raise FileExistsError(f'outfile {outfile} already exists')

        self.dpi = dpi
        self.fps = fps
        self.outfile = outfile
        self.logfile = logfile

        self.frame_size: typing.Tuple[float, float] = frame_size
        self.ffmpeg_proc: typing.Optional[mp.Process] = None
        self.logfh: typing.Optional[typing.TextIO] = None

        self.next_frame: int = 0
        self.ooo_frames: typing.Dict[str, bytes] = {}

        self.receive_queues: typing.List[mp.Queue] = []

    def __del__(self):
        if self.ffmpeg_proc is not None:
            self.ffmpeg_proc.kill()
            if self.logfh is None:
                self._create_loghandle()
            self._cleanup_ffmpeg()
        if self.logfh is not None:
            self._cleanup_logfh()

    def _spawn_ffpmeg(self):
        """Spawns the ffmpeg process
        """
        if self.ffmpeg_proc is not None:
            raise RuntimeError(f'_spawn_ffmpeg called when ffmpeg_proc is not None (is {self.ffmpeg_proc})')

        ffmpeg_path = str(mpl.rcParams['animation.ffmpeg_path'])

        bitrate = mpl.rcParams['animation.bitrate']

        args = [ffmpeg_path, '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{self.frame_size[0]}x{self.frame_size[1]}',
                '-pix_fmt', 'rgba', '-r', str(self.fps),
                '-loglevel', 'quiet',
                '-i', 'pipe:0',
                '-vcodec', 'h264', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart']

        if bitrate > 0:
            args.extend(['-b', '%dk' % bitrate])
        args.extend(['-y', self.outfile])

        self.ffmpeg_proc = sp.Popen(
            args, shell=False, stdout=None, stderr=None,
            stdin=sp.PIPE, creationflags=subprocess_creation_flags)

    def _cleanup_ffmpeg(self):
        """Cleans up the ffmpeg process. This will wait for it to terminate"""
        out, errs = self.ffmpeg_proc.communicate()
        if self.logfh is not None:
            if out is not None:
                if isinstance(out, bytes):
                    out = out.decode('utf-8')
                self.logfh.write(out + os.linesep)
            if errs is not None:
                if isinstance(errs, bytes):
                    errs = errs.decode('utf-8')
                self.logfh.write(errs + os.linesep)
        self.ffmpeg_proc = None

    def _create_loghandle(self):
        """Opens the logfile handle"""
        if self.logfh is not None:
            raise RuntimeError(f'_create_loghandle called when logfh is not None (is {self.logfh})')

        self.logfh = open(self.logfile, 'a')

    def _cleanup_logfh(self):
        """Closes the logfile handle"""
        if self.logfh is None:
            raise RuntimeError(f'_cleanup_logfh called when logfh is None')
        self.logfh.close()
        self.logfh = None

    def _cleanup_rec_queues(self):
        """Cleans up the receive queues"""
        self.receive_queues = []


    def start(self):
        """Sets this up for animating from the receive queues.
        """
        self._spawn_ffpmeg()

    def register_queue(self, queue: mp.Queue):
        """Registers the specified queue as one that will send us frames.

        Args:
            queue (mp.Queue): the queue which will send us frames
        """
        if queue is None:
            raise ValueError(f'expected queue is mp.Queue, got None')
        if not hasattr(queue, 'get_nowait'):
            raise ValueError(f'expected queue is mp.Queue, got {queue} (type={type(queue)})')
        self.receive_queues.append(queue)

    def check_queues(self):
        """Pushes queues onto the local memory stack. This operation should be fairly quick.
        """

        did_work = False

        for queue in self.receive_queues:
            if not queue.empty():
                did_work = True
                msg = queue.get_nowait()
                if not isinstance(msg, tuple):
                    raise ValueError(f'expected msg from receive_queue is tuple, got {msg} (type={type(msg)})')
                if len(msg) != 2:
                    raise ValueError(f'expected msg from receive_queue is (frame, raw) but got {msg} (len = {len(msg)} != 2)')
                if not isinstance(msg[0], int):
                    raise ValueError(f'expected msg from receive_queue is (frame, raw) but msg={msg} (type(msg[0]) = {type(msg[0])} != int)')
                if not isinstance(msg[1], bytes):
                    raise ValueError(f'expected msg from receive_queue is (frame, raw) but msg={msg} (type(msg[1]) = {type(msg[1])} != bytes)')

                frame = msg[0]
                img_bytes = msg[1]

                if frame < self.next_frame:
                    raise ValueError(f'expected msg from receive_queue is (frame, raw) but we got frame={frame} which we have already seen (next frame is {self.next_frame})')
                if frame in self.ooo_frames:
                    raise ValueError(f'expected msg from receive_queue is (frame, raw) but we already have frame={frame} in out of order frames')
                if len(self.ooo_frames) >= 5000:
                    raise ValueError(f'exceeded maximum frame cache (have {len(self.ooo_frames)} out of order while waiting for {self.next_frame})')
                self.ooo_frames[frame] = img_bytes

        return did_work

    def process_frame(self) -> bool:
        """Processes the next frame if it is available. Returns True if we
        processed a frame, False if we did not"""
        if self.next_frame not in self.ooo_frames:
            return False

        img_bytes = self.ooo_frames[self.next_frame]
        del self.ooo_frames[self.next_frame]

        block_size = 1024*4
        for kb_start in range(0, len(img_bytes), block_size):
            self.ffmpeg_proc.stdin.write(img_bytes[kb_start:kb_start+block_size])

        self.next_frame += 1
        return True

    def do_work(self):
        """A general catch-all function to do some work, returns True if did work"""
        did_work = False
        if self.check_queues():
            did_work = True
        if self.process_frame():
            did_work = True
        return did_work

    def finish(self):
        """Cleanly closes handles"""
        if self.ffmpeg_proc is not None:
            if self.logfh is None:
                self._create_loghandle()
            self._cleanup_ffmpeg()
            self._cleanup_logfh()
