"""Contains some additional nonlinearity like callables"""

import torch

def linear(acts: torch.tensor):
    """Does nothing"""
    return acts

def cube(acts: torch.tensor):
    """Cubes the values. The network could use this for expansion or compression
    if it has a linear layer preceeding it."""
    return acts ** 3

LOOKUP = {
    'relu': torch.relu,
    'tanh': torch.tanh,
    'none': linear,
    'linear': linear,
    'cube': cube
}