"""This module adds absolute norm layers which are meant to work in conjunction with batch norm
layers to allow a network which whitens between layers to still be evaluable when you don't have
a batch to calculate the sample mean and variance from. Note that while training you would still
want a batch norm so that the gradients reflect that the values are enforced to be zero-mean and
unit-variance, but this is not a necessary feature when evaluating the network.
"""

import torch

import shared.typeutils as tus

class EvaluatingAbsoluteNormLayer(torch.nn.Module):
    """A very simple 1-to-1 layer which subtracts of the precomputed mean and divides
    by the standard deviation

    Attributes:
        features (int): the number of features this layer is responsible for
        means (tensor[features]): the means to subtract of
        inv_std (tensor[features]): the reciprical of the standard deviation
    """
    def __init__(self, features: int, means: torch.tensor, inv_std: torch.tensor):
        super().__init__()
        tus.check(features=(features, int))
        tus.check_tensors(
            means=(means, (('features', features),), (torch.float, torch.double)),
            inv_std=(inv_std, (('features', features),), means.dtype)
        )
        self.features = features
        self.means = means
        self.inv_std = inv_std

    def forward(self, inp: torch.tensor):  # pylint: disable=arguments-differ
        tus.check_tensors(
            inp=(inp, (('batch', None), ('features', self.features)), self.means.dtype)
        )
        return (inp - self.means) * self.inv_std

    def to_linear(self) -> torch.nn.Linear:
        """Creates an equivalent version of this layer using torch.nn.Linear. It is much
        less memory efficient.

        Our version:
            a = (a - a_mean) / a_std

        Converting to linear:
            a = a / a_std - a_mean / a_std
        """
        lyr = torch.nn.Linear(self.features, self.features)
        lyr.weight.requires_grad = False
        lyr.bias.requires_grad = False

        lyr.weight.data[:] = 0
        lyr.weight.data[torch.eye(self.features, dtype=torch.uint8)] = self.inv_std
        lyr.bias.data[:] = -self.means * self.inv_std
        return lyr

    @classmethod
    def create_identity(cls, features: int, dtype=torch.float):
        """Creates an identity layer (does nothing) with the given number of features. Helpful
        when you are first initializing the network"""
        return cls(features, torch.zeros(features, dtype=dtype), torch.ones(features, dtype=dtype))

class LearningAbsoluteNormLayer(torch.nn.Module):
    """A tracking layer, not meant to be trainable via back propagation. Instead, this simply
    tracks all of its inputs using the method described at https://math.stackexchange.com/q/116344,
    with the modification that we store the first 100 values in memory to get a better estimate of
    the mean in the beginning.

    Note that this layer acts as the identity. The technique used to get the true norm is:

    1. put network in evaluation mode (freeze weights)
    2. push learning layers prior to the batch norm layers
    3. fetch many samples so that the learning layers learn
    4. replace batch norm with evaluating norm layers
    5. repeat the following until convergence:
        a. reset learning layers
        b. fetch many samples
        c. replace evaluating norm layers with the new learned ones

    Attributes:
        features (int): the number of features that we are expecting
        batch_size (int): how many samples we combine into a single tensor

        history (list[tensor[batch_size, features]]): the uncombined tensors
            that we have filled up already
        current (tensor[batch_size, features]): the current tensor we are trying
            to fill up with samples
        current_len (int): where we have filled up to in current
    """

    def __init__(self, features: int, batch_size: int = 1024):
        super().__init__()
        tus.check(features=(features, int), batch_size=(batch_size, int))
        self.features = features
        self.batch_size = batch_size

        self.history = []
        self.current = None
        self.current_len = 0

    def to_evaluative(self, like_batchnorm=False) -> EvaluatingAbsoluteNormLayer:
        """Gets the fixed absolute norm layer using the current estimate of mean and variance.
        Requires we have seen at least num_initial values.

        Args:
            like_batchnorm (bool): if True, 1e-5 is added to the variance before taking the sqrt
                to determine the standard deviation. This happens by default for the batch norm
                for numerical stability
        """
        if self.current is None or (not self.history and self.current_len < 3):
            raise ValueError('have not seen enough data to determine mean and std')

        all_data: torch.tensor = None
        if self.history:
            if self.current_len > 0:
                all_data = torch.cat(self.history + [self.current[:self.current_len]], dim=0)
            else:
                all_data = torch.cat(self.history, dim=0)
        else:
            all_data = self.current[:self.current_len]

        means = all_data.mean(dim=0)
        var = all_data.var(dim=0)

        std = var.sqrt() if not like_batchnorm else (var + 1e-5).sqrt()

        return EvaluatingAbsoluteNormLayer(self.features, means, 1 / std)

    def forward(self, inps: torch.tensor):  # pylint: disable=arguments-differ
        """Updates the running mean and variance using the given inputs"""
        tus.check_tensors(
            inps=(
                inps,
                (
                    ('batch', (self.current.dtype if self.current is not None else None)),
                    ('features', self.features)
                ),
                (torch.float, torch.double)
            )
        )
        if hasattr(inps, 'data'):
            inps = inps.data

        if self.current is None:
            self.current = torch.zeros((self.batch_size, self.features), dtype=inps.dtype)
            self.current_len = 0
        if self.current_len + inps.shape[0] == self.batch_size:
            self.current[self.current_len:] = inps
            self.history.append(self.current)
            self.current = torch.zeros_like(self.history[-1])
            self.current_len = 0
        elif self.current_len + inps.shape[0] < self.batch_size:
            self.current[self.current_len:self.current_len + inps.shape[0]] = inps
            self.current_len += inps.shape[0]
        else:
            split_at = self.batch_size - self.current_len
            self.forward(inps[:split_at])
            self.forward(inps[split_at:])
