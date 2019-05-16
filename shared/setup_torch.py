"""This module sets up torch, preventing it from sabotoging itself"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTOBLAS_MAIN_FREE'] = '1'

import torch
torch.set_num_threads(1)

import numpy as np
np.seterr('raise')

import multiprocessing as mp
try:
    mp.set_start_method('spawn')
    print('successfully set start method for threads to new process')
except RuntimeError:
    print('failed to set multiprocessing spawn method; this happens on windows')