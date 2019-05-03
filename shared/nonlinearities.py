"""Contains some additional nonlinearity like callables"""

import torch
import torch.autograd

def linear(acts: torch.tensor):
    """Does nothing"""
    return acts

def cube(acts: torch.tensor):
    """Cubes the values. The network could use this for expansion or compression
    if it has a linear layer preceeding it."""
    return acts ** 3

def tanh_recip(acts: torch.tensor):
    """Takes the hyperbolic tangent of the reciprical
    """
    return torch.tanh(torch.reciprocal(acts))

class ISRLU(torch.autograd.Function):
    """Inverse square root linear unit: https://arxiv.org/abs/1710.09967

    Improving Deep Learning by Inverse Square Root Linear Units (ISRLUs)
    Brad Carlile, Guy Delamarter, Paul Kinney, Akiko Marti, Brian Whitney
    (2017)
    """

    @staticmethod
    def forward(ctx, tensor, alpha=1): #pylint: disable=arguments-differ
        """Calculates the forward pass for an ISRLU unit"""
        negatives = torch.min(tensor, torch.Tensor([0]))
        nisr = torch.rsqrt(1. + alpha * (negatives ** 2))
        return tensor * nisr

LOOKUP = {
    'relu': torch.relu,
    'tanh': torch.tanh,
    'tanh_recip': tanh_recip,
    'none': linear,
    'linear': linear,
    'cube': cube,
    'isrlu': ISRLU.apply
}
