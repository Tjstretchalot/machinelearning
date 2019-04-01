"""This module contains an RNN torch.nn.Module and related"""

import typing
import torch
import torch.nn

import shared.weight_inits as wi
from shared.models.generic import Network
from shared.teachers import NetworkTeacher

class RNNHiddenActivations(typing.NamedTuple):
    """This is the named tuple which is passed to the hidden activations
    callback, if provided

    Attributes:
        recur_step (int): a number which is the number of times we have gone through
            the hidden weights. Will be 0 on the first call (just gone through the
            input weights + biases), then end after the recur_times

        hidden_acts ([tensor, batch_size x hidden_dim]): the hidden activations in
            the network. This is a parameter and thus must be detach()'d to save
    """
    recur_step: int
    hidden_acts: torch.tensor

class NaturalRNN(Network):
    """
    Describes a simple non-spiking recurrent neural network. It is natural in the sense that all of
    the relevant variables are easily available as attributes. It's a programming-friendly
    version of the default torch RNN, which has strange variable names and layer-code.

    Attributes:
        hidden_dim (int): the dimension of the hidden activations, i.e., the number of hidden nodes

        nonlinearity (callable): the nonlinearity used to go from inputs to activations

        input_weights (Parameter[tensor, input_dim x hidden_dim])
        input_biases (Parameter[tensor, hidden_dim])

        hidden_weights (Parameter[tensor, hidden_dim x hidden_dim])
        hidden_biases (Parameter[tensor, hidden_dim])

        output_weights (Parameter[tensor, hidden_dim x output_dim])
        output_biases (Parameter[tensor, output_dim])
    """

    def __init__(self, nonlinearity: typing.Callable, input_weights: torch.tensor,
                 input_biases: torch, hidden_weights: torch.tensor, hidden_biases: torch.tensor,
                 output_weights: torch.tensor, output_biases: torch.tensor):
        super().__init__(input_weights.shape[0], output_weights.shape[1])

        self.hidden_dim = input_weights.shape[1]

        self.nonlinearity = nonlinearity
        self.input_weights = torch.nn.Parameter(input_weights, requires_grad=False)
        self.input_biases = torch.nn.Parameter(input_biases, requires_grad=True)
        self.hidden_weights = torch.nn.Parameter(hidden_weights, requires_grad=True)
        self.hidden_biases = torch.nn.Parameter(hidden_biases, requires_grad=True)
        self.output_weights = torch.nn.Parameter(output_weights, requires_grad=True)
        self.output_biases = torch.nn.Parameter(output_biases, requires_grad=True)

    @classmethod
    def create(
            cls, nonlinearity: str, input_dim: int, hidden_dim: int, output_dim: int,
            input_weights: wi.WeightInitializer, input_biases: wi.WeightInitializer,
            hidden_weights: wi.WeightInitializer, hidden_biases: wi.WeightInitializer,
            output_weights: wi.WeightInitializer, output_biases: wi.WeightInitializer
        ) -> 'NaturalRNN':
        """An alternative constructor which can be called more easily from a serialized dictionary.
        Each of the wi.WeightInitializer's are runtime type-checked and, if they fail, are sent
        through wi.deserialize which will error on failure.

        Arguments:
            nonlinearity (str): either 'tanh' or 'relu'
            input_dim (int): the dimension of the inputs
            hidden_dim (int): the dimension of the hidden activations, i.e., number of hidden nodes
            output_dim (int): the dimension of the outputs, i.e., number of labels

            The remaining are either wi.WeightInitializer or something that can be passed to
            wi.deserialize

        Returns:
            The newly constructed NaturalRNN
        """

        if nonlinearity == 'tanh':
            _nonlinearity = torch.tanh
        elif nonlinearity == 'relu':
            _nonlinearity = torch.relu
        else:
            raise ValueError(f'bad nonlinearity: expected \'tanh\' or \'relu\' but got {nonlinearity}')

        _input_weights = torch.zeros((input_dim, hidden_dim), dtype=torch.double)
        wi.deser_or_noop(input_weights).initialize(_input_weights)

        _input_biases = torch.zeros(hidden_dim, dtype=torch.double)
        wi.deser_or_noop(input_biases).initialize(_input_biases)

        _hidden_weights = torch.zeros((hidden_dim, hidden_dim), dtype=torch.double)
        wi.deser_or_noop(hidden_weights).initialize(_hidden_weights)

        _hidden_biases = torch.zeros(hidden_dim, dtype=torch.double)
        wi.deser_or_noop(hidden_biases).initialize(_hidden_biases)

        _output_weights = torch.zeros((hidden_dim, output_dim), dtype=torch.double)
        wi.deser_or_noop(output_weights).initialize(_output_weights)

        _output_biases = torch.zeros(output_dim, dtype=torch.double)
        wi.deser_or_noop(output_biases).initialize(_output_biases)

        return cls(_nonlinearity, _input_weights, _input_biases, _hidden_weights, _hidden_biases,
                   _output_weights, _output_biases)

    def forward(self, *args) -> torch.tensor:
        if not args:
            raise ValueError('must get batched points (tensor batch_size x input_size) as input!')

        batched_pts = args[0]
        if not torch.is_tensor(batched_pts):
            raise ValueError('batched_pts should be a tensor (batch_size x input_size)'
                             + f', got not a tensor: {batched_pts}')

        if len(batched_pts.shape) != 2 or batched_pts.shape[1] != self.input_dim:
            raise ValueError('batched_pts should have shape (batch_size x input_size)'
                             + f', but is {batched_pts.shape}')

        recur_times = 1
        if len(args) > 1:
            recur_times: int = args[1]
            if not isinstance(recur_times, int):
                raise ValueError(f'recur_times should be an int, but is {recur_times}')

        hidden_acts_callback = None
        if len(args) > 2:
            hidden_acts_callback = args[2]
            if hidden_acts_callback is not None and not callable(hidden_acts_callback):
                raise ValueError('hidden_acts_callback should be callable or None'
                                 + f', but is {hidden_acts_callback}')

        input_times = 1
        if len(args) > 3:
            input_times: int = args[3]
            if not isinstance(input_times, int):
                raise ValueError(f'input_times should be an int, but is {input_times}')

        return self._typed_forward(batched_pts, recur_times, hidden_acts_callback, input_times)

    def _typed_forward(self, batched_points: torch.tensor, recur_times: int,
                       hidden_acts_callback: typing.Callable, input_times: int) -> torch.tensor:
        batch_size = batched_points.shape[0]

        inp_currents = (
            batched_points @ self.input_weights + self.input_biases.expand(batch_size, -1))
        hidden_biases_expanded = self.hidden_biases.expand(batch_size, -1)
        hidden_acts = self.nonlinearity(inp_currents + hidden_biases_expanded)

        if hidden_acts_callback:
            hidden_acts_callback(RNNHiddenActivations(recur_step=0, hidden_acts=hidden_acts))

        for recur_time in range(1, recur_times + 1):
            currents = hidden_acts @ self.hidden_weights + hidden_biases_expanded
            if recur_time < input_times:
                currents = currents + inp_currents
            hidden_acts = self.nonlinearity(currents)

            if hidden_acts_callback:
                hidden_acts_callback(
                    RNNHiddenActivations(recur_step=recur_time, hidden_acts=hidden_acts))

        return (
            self.nonlinearity(
                hidden_acts @ self.output_weights + self.output_biases.expand(batch_size, -1)
            )
        )

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
