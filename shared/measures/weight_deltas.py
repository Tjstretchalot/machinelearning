"""This is a through-time measure that attempts to quantify what the changes
in weights look like across time. This can help determine the type of randomness
that SGD creates
"""

from shared.trainer import GenericTrainingContext
from shared.npmp import NPDigestor
from shared.filetools import zipdir
import shared.measures.utils as mutils
import torch
import typing
import numpy as np
import matplotlib.pyplot as plt
import os

def _binned2norm(induced: np.ndarray, outpath: str, title: str, dpi=400, transparent=False):
    """The target for Binned2Norm: bar plot of the induced changes in the 2norm

    Args:
        induced (np.ndarray): a list of floats of induced changes in 2-norm
        outpath (str): the folder or zip file to save to
    """
    _, outfolder = mutils.process_outfile(outpath, False)
    os.makedirs(outfolder, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_xlabel('$Induced \\Delta \\| W \\|_2$').set_fontsize(16)
    ax.set_ylabel('Count').set_fontsize(16)

    ax.set_title(title).set_fontsize(18)

    ax.hist(induced, bins=10)

    fig.savefig(os.path.join(outfolder, 'histogram.png'), dpi=dpi, transparent=transparent)
    plt.close(fig)

    zipdir(outfolder)

class Binned2Norm:
    """This is a measure of weight changes produced by measuring the 2-norm of the
    change in a particular weight matrix induced by training. Then it bar-plots the
    data.

    Attributes:
        tensor_fetcher (callable(GenericTrainingContext) -> torch.tensor)
        dig (NPDigestor): the digestor for long-term operations
        outpath (str): the path to save to (should be a folder)
        title (str): the title for the plot
        last_val (torch.tensor): the value of the tensor when we last checked
        induced_norms (list[float]): a list of induced changes in the two-norm
    """

    def __init__(self, tensor_fetcher: typing.Callable, dig: NPDigestor, outpath: str, title: str):
        if not callable(tensor_fetcher):
            raise ValueError(f'expected tensor_fetcher is callable, got {tensor_fetcher} (type={type(tensor_fetcher)})')

        self.tensor_fetcher = tensor_fetcher
        self.dig = dig
        self.outpath = outpath
        self.title = title
        self.last_val = None
        self.induced_norms = []


    def setup(self, context: GenericTrainingContext, **kwargs) -> None:
        """Initializes the last val"""
        self.last_val = self.tensor_fetcher(context).clone()

    def post_train(self, context, loss):
        """Fetches the change in the relevant tensor"""
        cur_val = self.tensor_fetcher(context)
        self.induced_norms.append(float(torch.norm(self.last_val - cur_val)))
        self.last_val = cur_val.clone()

    def finished(self, context: GenericTrainingContext, result: dict):
        """Actually plots through the digestor"""
        context.logger.info('[WDS] Plotting Binned2Norm (title=%s) (outpath=%s)', self.title, str(self.outpath))
        ind_norm_np = np.array(self.induced_norms, dtype='float64')
        self.dig(target_module='shared.measures.weight_deltas', target_name='_binned2norm',
                 induced=ind_norm_np, outpath=self.outpath, title=self.title)


