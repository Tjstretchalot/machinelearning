# pylint: skip-file
"""This module is meant to act as a template for new measures."""

import torch
import typing
import numpy as np

import shared.measures.utils as mutils
import shared.npmp as npmp
from shared.pwl import PointWithLabelProducer

class MyTrajectory:
    pass

def measure(hacts: mutils.NetworkHiddenActivations) -> MyTrajectory:
    raise NotImplementedError()

def plot(traj: MyTrajectory, outfile: str, exist_ok: bool = False) -> None:
    outfile, outfile_wo_ext = mutils.process_outfile(outfile, exist_ok)
    raise NotImplementedError()

digest = mutils.digest_std(measure, plot)
during_training = mutils.during_training_std('TODO', measure, plot, 'shared.measures.TODO')