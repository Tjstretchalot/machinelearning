"""Additional criterions for neural networks
"""

import torch
import numpy as np

def meansqerr(output: torch.tensor, labels: torch.tensor):
    """This criterion evaluates a one-hot vector on mean-squared error when
    the labels are the indices of the output that should be 1. This is a drop-in
    replacement for cross-entropy loss whereas the default MSEError requires changing
    your encoding.

    Args:
        output (torch.tensor): the output
        labels (torch.tensor): the expected labels

    Returns:
        loss (torch.tensor): the loss of the network
    """
    bats = output.shape[0]

    adj_output = output.clone()
    adj_output[torch.arange(bats), labels] -= 1
    return torch.mean(adj_output ** 2)

def hubererr(output: torch.tensor, labels: torch.tensor):
    """This criterion is like meansqerr except uses huber loss instead of
    squared error.
    """
    bats = output.shape[0]
    target = torch.zeros_like(output)
    target[torch.arange(bats), labels] += 1
    return torch.functional.F.smooth_l1_loss(output, target)

def create_meansqerr_regul(noise_strength=0):
    """This returns a which criterion evaluates a one-hot vector on mean-squared error
    with a regularizing term on the two-norm of the output"""

    def result(output: torch.tensor, labels: torch.tensor):
        bats = output.shape[0]
        adj_output = output.clone()
        adj_output[torch.arange(bats), labels] -= 1

        loss_mse = torch.mean(adj_output ** 2)
        loss_noise = (np.pi*noise_strength*noise_strength)/2 * torch.sum(output ** 2)

        return loss_mse + loss_noise

    return result