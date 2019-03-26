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

class RNNTeacher(NetworkTeacher):
    """Describes something which can teach an rnn

    Attributes:
        recurrent_times (int): how many times to recur
        input_times (int): how many times to present the input
    """

    def __init__(self, recurrent_times: int, input_times: int = 1):
        self.recurrent_times = recurrent_times
        self.input_times = input_times

    def teach_many(self, network: Network, optimizer: torch.optim.Optimizer, criterion: typing.Any,
                   points: torch.tensor, labels: torch.tensor):
        network.train()
        with torch.set_grad_enabled(True):
            network.zero_grad()
            optimizer.zero_grad()
            result = network(points, self.recurrent_times, None, self.input_times)
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()
        return loss.item()

    def classify_many(self, network: Network, points: torch.tensor, out: torch.tensor):
        network.eval()
        with torch.no_grad():
            result = network(points, self.recurrent_times, None, self.input_times)
        out.copy_(result)