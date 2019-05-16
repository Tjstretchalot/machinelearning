"""This module sets up torch, preventing it from sabotoging itself"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTOBLAS_MAIN_FREE'] = '1'

import torch
import time
import random
torch.set_num_threads(1)
torch.manual_seed(int(time.time() * 1000))
torch.randint(1, 100, (1000,))
torch.manual_seed(int(torch.randint(1, 2**16, (1,)).item() * time.time()))

torch.randint(1, 100, (1000,))
torch.randint(1, 100, (int(torch.randint(1, 100, (1,)).item()),))

random.seed()

import numpy as np
np.seterr('raise')
np.random.seed()

import multiprocessing as mp
try:
    mp.set_start_method('spawn')
    print('successfully set start method for threads to new process')
except RuntimeError:
    pass