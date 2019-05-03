"""This runner just checks that the ISRLU nonlinearity is working"""

import shared.nonlinearities as snonlin
import torch
import numpy as np

ALPHA = 1
def func(x, alpha=ALPHA):
    """elementwise isrlu"""
    if x > 0:
        return x
    return x / (np.sqrt(1 + alpha * x * x))

def deriv(x, alpha=ALPHA):
    """elementwise derivative"""
    if x > 0:
        return 1
    val = 1 / (np.sqrt(1 + alpha * x * x))
    return val * val * val

class Context:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

def random_test():
    """tests the nonlin against random values"""
    ctx = Context()
    test_data = torch.randn(10, 10)

    forwarded = snonlin.ISRLU.forward(ctx, test_data, alpha=ALPHA)
    for i in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            expected = func(test_data[i, j].item())
            actual = forwarded[i, j].item()
            if np.abs(expected - actual) > 1e-6:
                raise ValueError(f'bad forward implementation; func({test_data[i, j].item()}) should be {expected} but got {actual}')

    grad_output = torch.randn(10, 10)
    backwarded = snonlin.ISRLU.backward(ctx, grad_output)
    for i in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            expected = grad_output[i, j] * deriv(test_data[i, j].item())
            actual = backwarded[i, j].item()
            if np.abs(expected - actual) > 1e-6:
                raise ValueError(f'bad backward implementation; deriv({test_data[i, j].item()}) should be {expected} but got {actual}')

    print('random test passed')

if __name__ == '__main__':
    random_test()