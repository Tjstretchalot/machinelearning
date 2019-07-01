"""This module adds absolute norm layers which are meant to work in conjunction with batch norm
layers to allow a network which whitens between layers to still be evaluable when you don't have
a batch to calculate the sample mean and variance from. Note that while training you would still
want a batch norm so that the gradients reflect that the values are enforced to be zero-mean and
unit-variance, but this is not a necessary feature when evaluating the network.
"""

import torch

import shared.typeutils as tus

class EvaluatingAbsoluteNormLayer:
    """A very simple 1-to-1 layer which subtracts of the precomputed mean and divides
    by the standard deviation

    Attributes:
        features (int): the number of features this layer is responsible for
        means (tensor[features]): the means to subtract of
        inv_std (tensor[features]): the reciprical of the standard deviation
    """
    def __init__(self, features: int, means: torch.tensor, inv_std: torch.tensor):
        tus.check(features=(features, int))
        tus.check_tensors(
            means=(means, (('features', features),), (torch.float, torch.double)),
            inv_std=(inv_std, (('features', features),), means.dtype)
        )
        self.features = features
        self.means = means
        self.inv_std = inv_std

    def __call__(self, inp: torch.tensor):
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
    def create_identity(cls, features: int, dtype = torch.float):
        """Creates an identity layer (does nothing) with the given number of features. Helpful
        when you are first initializing the network"""
        return cls(features, torch.zeros(features, dtype=dtype), torch.ones(features, dtype=dtype))

class LearningAbsoluteNormLayer:
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
        num_initial (int): the number of samples we store before in memory before the first
            estimate of mean and variance. used to speed up convergence

        means (tensor[features], optional): the current estimate for the mean value for each
            feature. initialized after we have filled the initial tensor
        variances (tensor[features], optional): the current estimate for variance for each
            feature. initialized after we have filled the initial tensor
        num_seen (int): the number of samples we have seen so far

        initial (tensor[num_initial, features], optional): a store for the first set of values
            seen, initialized on the first set of samples and uninitialized once filled and used
            to calculate the first set of means and variances
    """
    def __init__(self, features: int, num_initial: int = 100):
        tus.check(features=(features, int), num_initial=(num_initial, int))
        self.features = features
        self.num_initial = num_initial

        self.means = None
        self.variances = None
        self.num_seen = 0

        self.initial = None

    def to_evaluative(self) -> EvaluatingAbsoluteNormLayer:
        """Gets the fixed absolute norm layer using the current estimate of mean and variance.
        Requires we have seen at least num_initial values.
        """
        if self.num_seen < self.num_initial:
            raise ValueError('have not seen enough data to convert to evaluative '
                             + f'({self.num_seen}/{self.num_initial})')
        means = self.means.clone()
        inv_std = (1 / self.variances.clone()).sqrt()
        return EvaluatingAbsoluteNormLayer(self.features, means, inv_std)

    def __call__(self, inps: torch.tensor):
        """Updates the running mean and variance using the given inputs"""
        tus.check_tensors(
            inps=(inps, (('batch', None), ('features', self.features)),
                  (torch.float, torch.double))
        )
        inps = inps.detach()
        if self.num_seen < self.num_initial:
            if self.initial is None:
                self.initial = torch.zeros((self.num_initial, self.features), dtype=inps.dtype)
            elif inps.dtype != self.initial.dtype:
                raise ValueError(f'dtype of inps changed from {self.initial.dtype} to {inps.dtype}')

            if self.num_seen + inps.shape[0] <= self.num_initial:
                self.initial[self.num_seen:self.num_seen + inps.shape[0]] = inps
                self.num_seen += inps.shape[0]

                if self.num_seen == self.num_initial:
                    self._initial_filled()
                return

            num_for_initial = self.num_initial - self.num_seen - inps.shape[0]
            self(inps[:num_for_initial])
            self(inps[num_for_initial:])
            return

        prev_means = self.means.clone()
        for i in range(inps.shape[0]):
            self.means += (inps[i] - self.means) / (self.num_seen + 1)
            vari_sum = (inps[i] - prev_means) * (inps[i] - self.means)  # this could be negative
            self.variances += vari_sum.abs()
            prev_means[:] = self.means
            self.num_seen += 1

    def _initial_filled(self):
        """We invoke this when we fill the initial tensor"""
        self.means = self.initial.mean(dim=0)
        self.variances = self.initial.var(dim=0)

        self.initial = None
