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
        negatives = torch.min(tensor, torch.tensor((0,), dtype=tensor.dtype))
        nisr = torch.rsqrt(1. + alpha * (negatives ** 2))
        ctx.save_for_backward(nisr)
        return tensor * nisr

    @staticmethod
    def backward(ctx, grad_output):
        nisr, = ctx.saved_tensors
        return grad_output * (nisr.clone() ** 3)

LOOKUP = {
    'relu': torch.relu,
    'tanh': torch.tanh,
    'tanh_recip': tanh_recip,
    'none': linear,
    'linear': linear,
    'cube': cube,
    'isrlu': ISRLU.apply
}

EXTENDED_LOOKUP = set(k for k in LOOKUP.keys())
EXTENDED_LOOKUP.add('leakyrelu')

def extended_lookup(val):
    if val in LOOKUP:
        return LOOKUP[val]
    if val == 'leakyrelu':
        return torch.nn.LeakyReLU()
    raise ValueError(f'unknown nonlinearity: {val}')