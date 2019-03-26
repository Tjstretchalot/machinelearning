"""Describe a generic model. Bakes in a few more variables that cannot be reasonably split from
the underlying model."""

import torch.nn

class Network(torch.nn.Module):
    """Describes a generic network.

    Attributes:
        input_dim (int): the number of dimensions in the input space
        output_dim (int): the number of dimensions in the output space
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, *args):
        raise NotImplementedError()