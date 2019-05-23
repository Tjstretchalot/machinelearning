"""Contains the generic description of a network teacher and some implementations"""

import typing
import torch

from shared.models.generic import Network
from shared.seqseqprod import Sequence
from shared.perf_stats import PerfStats, NoopPerfStats

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

class SeqSeqTeacher:
    """Describes something that can teach sequence-sequence models"""

    def teach_many(self, network: Network, optimizers: typing.List[torch.optim.Optimizer],
                   criterion: typing.Any, inputs: typing.List[Sequence],
                   outputs: typing.List[Sequence],
                   perf_stats: PerfStats = NoopPerfStats()) -> float:
        """Teaches the specified network by feeding it sequences and their associated expected
        outputs.

        Arguments:
            network (Network): the network to teach
            optimizers (torch.optim.Optimizer): the optimizers to use. The number of meaning depend
                on the network
            criterion (torch.nn.loss._Loss): the loss function
            inputs (list[Sequence]): a list of length batch_size that contains the sequences
                to send to the network
            outputs (list[Sequence]): a list of length batch_size that contains the sequences
                we expect from the network
            perf_stats (PerfStats, optional): used to track performance information. Starts
                entering subdivisions of teachmany

        Returns:
            the average loss on the batch
        """

        raise NotImplementedError()

    def classify_many(self, network: Network,
                      inputs: typing.List[Sequence],
                      perf_stats: PerfStats = NoopPerfStats()) -> typing.List[Sequence]:
        """Runs the input through the network and returns the result

        Arguments:
            network (Network): the network to make transform/classify
            inputs (list[Sequence]): a list of length batch_size that contains the sequences
                to send to the network
            perf_stats (PerfStats, optional): used to store performance information. starts
                by entering subdivisions of classify_many

        Returns:
            a list of length batch_size that contains the sequences the network produced
        """

        raise NotImplementedError()
