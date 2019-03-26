"""This module sets up torch, preventing it from sabotoging itself"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTOBLAS_MAIN_FREE'] = '1'

import torch
torch.set_num_threads(1)
