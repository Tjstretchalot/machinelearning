"""Utils regarding cloning / copying things"""

import torch

def copy_linear(lyr: torch.nn.Linear) -> torch.nn.Linear:
    cp_lyr = torch.nn.Linear(lyr.in_features, lyr.out_features, lyr.bias is not None)

    cp_lyr.weight.data[:] = lyr.weight.data
    if lyr.bias is not None:
        cp_lyr.bias.data[:] = lyr.bias.data
    return cp_lyr
