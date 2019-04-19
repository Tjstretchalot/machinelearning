"""Additional criterions for neural networks
"""

import torch

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