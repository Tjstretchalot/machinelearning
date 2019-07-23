"""A simple affine layer, which performs a linear transformation y = a*x + b to each
feature. Helpful when you want a batch norm layer but don't want to tie the affine
transformation to it"""

import torch
import pytypeutils as tus

class AffineLayer(torch.nn.Module):
    """Performs an affine transformation on each feature

    Attributes:
        features (int): the number of features
        mult (torch.tensor[features]): the multiplicative terms
        add (torch.tensor[features]): the additive terms
    """

    def __init__(self, features: int, mult: torch.tensor, add: torch.tensor):
        super().__init__()
        tus.check(features=(features, int))
        tus.check_tensors(
            mult=(mult, (('features', features),), (torch.float, torch.double)),
            add=(add, (('features', features),), mult.dtype)
        )
        self.features = features
        self.mult = torch.nn.Parameter(mult)
        self.add = torch.nn.Parameter(add)

    @classmethod
    def create(cls, features: int, dtype=torch.float):
        """Creates an identity transformation with the given number of features"""
        return cls(features, torch.ones(features, dtype=dtype), torch.zeros(features, dtype=dtype))

    def forward(self, inp: torch.tensor): # pylint: disable=arguments-differ
        return inp * self.mult + self.add
