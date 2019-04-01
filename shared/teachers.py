"""Contains the generic description of a network teacher and some implementations"""

import typing
import torch

from shared.models.generic import Network

class NetworkTeacher:
    """Describes something that can send points to a particular network."""

    def teach_many(self, network: Network, optimizer: torch.optim.Optimizer, criterion: typing.Any,
                   points: torch.tensor, labels: torch.tensor) -> float:
        """Teaches the specified network by feeding it the given points and their associated
        labels.

        Arguments:
            network (Network): the network to teach
            optimizer (torch.optim.Optimizer): the optimizer to use
            criterion (torch.nn.loss._Loss): the loss
            points (torch.tensor [batch_size x input_dim]): the points to send to the network
            labels (torch.tensor [batch_size]): the correct labels for the points
        """

        raise NotImplementedError()

    def classify_many(self, network: Network, points: torch.tensor, out: torch.tensor):
        """Classifies the gives points into the given tensor.

        Arguments:
            network (Network): the network to make classify
            points (torch.tensor [batch_size x input_dim]): the points to send to the network
            out (torch.tensor [batch_size]): where to save the output
        """

        raise NotImplementedError()
