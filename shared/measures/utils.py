"""A collection of utility functions for measures"""

import typing
import torch
import numpy as np
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations
from shared.pwl import PointWithLabelProducer

def verify_points_and_labels(sample_points: torch.tensor, sample_labels: torch.tensor):
    """Verifies that the given sample points and labels make sense together

    Args:
        sample_points (torch.tensor): the points received
        sample_labels (torch.tensor): the labels you received
    """
    if not torch.is_tensor(sample_points):
        raise ValueError(f'expected sample_points is tensor, got {sample_points} (type={type(sample_points)})')
    if len(sample_points.shape) != 2:
        raise ValueError(f'expected sample_points has shape [num_pts, input_dim] but has shape {sample_points.shape}')
    if sample_points.dtype not in (torch.float, torch.double):
        raise ValueError(f'expected sample_points is float-like but has dtype={sample_points.dtype}')
    if not torch.is_tensor(sample_labels):
        raise ValueError(f'expected sample_labels is tensor, got {sample_labels} (type={type(sample_labels)})')
    if len(sample_labels.shape) != 1:
        raise ValueError(f'expected sample_labels has shape [num_pts] but has shape {sample_labels.shape}')
    if sample_labels.dtype not in (torch.uint8, torch.int, torch.long):
        raise ValueError(f'expected sample_labels is int-like but has dtype={sample_labels.dtype}')
    if sample_points.shape[0] != sample_labels.shape[0]:
        raise ValueError(f'expected sample_points has shape [num_pts, input_dim] and sample_labels has shape [num_pts] but sample_points.shape={sample_points.shape} and sample_labels.shape={sample_labels.shape} (mismatch on dim 0)')


class NetworkHiddenActivations:
    """Describes the result from get_hidacts style functions. Describes the hidden
    activations of a network through layers or through time, depending on if the
    network is feed forward or not.

    Attributes:
        netstyle (str): one of 'feedforward' or 'recurrent'
        sample_points (tensor[num_pts, input_dim], float-like):
            the points which were used to sample
        sample_labels (tensor[num_pts], int-like): the labels of the points

        hid_acts (list[num_layers of tensor[num_pts, layer_size], float-like):
            the activations of the network as you go through time. accessible through
            time or layers, depending on the netstyle
    """

    def __init__(self, netstyle: str, sample_points: torch.tensor, sample_labels: torch.tensor,
                 hid_acts: typing.List[torch.tensor]) -> None:
        if netstyle not in ('feedforward', 'recurrent'):
            raise ValueError(f'expected netstyle is \'feedforward\' or \'recurrent\', got {netstyle} (type={type(netstyle)})')
        verify_points_and_labels(sample_points, sample_labels)
        if not isinstance(hid_acts, list):
            raise ValueError(f'expected hid_acts is list[tensor[num_pts, layer_size]] but is {hid_acts} (type={type(hid_acts)})')
        for idx, lyr in enumerate(hid_acts):
            if not torch.is_tensor(lyr):
                raise ValueError(f'expected hid_acts[{idx}] is tensor[num_pts, layer_size] but is not tensor: {lyr} (type={type(lyr)})')
            if len(lyr.shape) != 2:
                raise ValueError(f'expected hid_acts[{idx}].shape is [num_pts, layer_size] but is {lyr.shape}')
            if lyr.shape[0] != sample_points.shape[0]:
                raise ValueError(f'expected hid_acts[{idx}].shape is [num_pts={sample_points.shape[0]}, layer_size] but is {lyr.shape}')
            if lyr.dtype not in (torch.float, torch.double):
                raise ValueError(f'expected hid_acts[{idx}] is float-like but dtype={lyr.dtype}')

        self.netstyle = netstyle
        self.sample_points = sample_points
        self.sample_labels = sample_labels
        self.hid_acts = hid_acts

    def numpy(self):
        """Changes this to numpy style from torch style. Always constructed torch-style
        """
        self.sample_points = self.sample_points.numpy()
        self.sample_labels = self.sample_labels.numpy()
        for i in range(len(self.hid_acts)):
            self.hid_acts[i] = self.hid_acts[i].numpy()
        return self

    def torch(self):
        """Changes this to torch style from numpy style. Always constructed torch-style
        """
        self.sample_points = torch.from_numpy(self.sample_points)
        self.sample_labels = torch.from_numpy(self.sample_labels)
        for i in range(len(self.hid_acts)):
            self.hid_acts[i] = torch.from_numpy(self.hid_acts[i])
        return self

    @property
    def num_pts(self):
        """Gets the number of points run through the network"""
        return self.sample_points.shape[0]

    @property
    def input_dim(self):
        """Gets the embedding space of the inputs"""
        return self.sample_points.shape[1]

    @property
    def num_layers(self):
        """Gets the number of activations available. This is 1 + the number of layers in the
        network, since hid_acts[0] is the input space. This distinction is not really important
        when plotting since it could alternatively be though of as the index of the layer which
        you want activation prior to applying, and that makes sense for 0-number of layers in the
        network (inclusive)"""
        return len(self.hid_acts)

def get_hidacts_ff_with_sample(network: FeedforwardNetwork, sample_points: torch.tensor,
                               sample_labels: torch.tensor) -> NetworkHiddenActivations:
    """Gets the hidden activations for a feedforward network when running the given sample
    points through it, at attaches the given sample labels to the result.

    Args:
        network (FeedforwardNetwork): the network to forward through
        sample_points (torch.tensor): the points to forward through
        sample_labels (torch.tensor): the labels of the points which will be forwarded

    Returns:
        NetworkHiddenActivations: the activations of the network
    """

    if not isinstance(network, FeedforwardNetwork):
        raise ValueError(f'expected network is FeedforwardNetwork, got {network} (type={type(network)})')
    verify_points_and_labels(sample_points, sample_labels)

    hid_acts = []

    def on_hidacts(acts_info: FFHiddenActivations):
        hid_acts.append(acts_info.hidden_acts.detach())

    network(sample_points, on_hidacts)
    return NetworkHiddenActivations('feedforward', sample_points, sample_labels, hid_acts)


def get_hidacts_ff(network: FeedforwardNetwork, pwl: PointWithLabelProducer,
                   num_points: typing.Optional[int] = None) -> NetworkHiddenActivations:
    """Creates a sample of at most num_points from the given point with label producer without
    affecting its internal state. Then runs those through the network and returns the hidden
    activations that came out of that

    Arguments:
        network (FeedforwardNetwork): the network which the points should be run through
        pwl (PointWithLabelProducer): where to acquire the points
        num_points (int, optional): if specified, the number of points to fetch. if not specified
            this is min(constant*network.num_layers, pwl.epoch_size) where the constant is
            reasonable of the order of 100
    """

    if num_points is None:
        num_points = min(50*network.output_dim, pwl.epoch_size)

    if not isinstance(network, FeedforwardNetwork):
        raise ValueError(f'expected network is FeedforwardNetwork, got {network} (type={type(network)})')
    if not isinstance(pwl, PointWithLabelProducer):
        raise ValueError(f'expected pwl is PointWithLabelProducer, got {pwl} (type={type(pwl)})')
    if not isinstance(num_points, int):
        raise ValueError(f'expected num_points is int, got {num_points} (type={type(num_points)})')

    sample_points = torch.zeros((num_points, pwl.input_dim), dtype=torch.double)
    sample_labels = torch.zeros(num_points, dtype=torch.uint8)

    pwl.mark()
    pwl.fill(sample_points, sample_labels)
    pwl.reset()
    return get_hidacts_ff_with_sample(network, sample_points, sample_labels)
