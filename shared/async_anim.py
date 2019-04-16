"""Provides tools to animate figures in push frame / pop frame style
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import writers

import logging


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
