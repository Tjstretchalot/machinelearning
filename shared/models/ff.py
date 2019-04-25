"""This contains various simple feed-forward models"""

import torch
import torch.nn

from shared.models.generic import Network
from shared.teachers import NetworkTeacher
import shared.weight_inits as wi
import shared.nonlinearities as nonlins

import typing
import math
from functools import reduce
import operator

class FFHiddenActivations(typing.NamedTuple):
    """Describes the feedforward activations within the network

    Attributes:
        layer (int): the layer counter, starting at 0
        hidden_acts (attached tensor, layer_input_size x layer_output_size)
    """
    layer: int
    hidden_acts: torch.tensor

class FeedforwardNetwork(Network):
    """A tag interface for feed-forward style networks. They will all accept (input,
    hidden_acts_callback) for their first two arguments, where hidden_acts_callback
    is passed just one argument, an FFHiddenActivations instance.

    Attributes:
        num_layers (int): how many layers are in this network
    """
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__(input_dim, output_dim)

        if not isinstance(num_layers, int):
            raise ValueError(f'expected num_layers is int, got {num_layers} (type={type(num_layers)})')
        self.num_layers = num_layers

    def forward(self, *args):
        raise NotImplementedError

class FeedforwardSmall(FeedforwardNetwork):
    """A small feedforward model for classification. Two nonlinear
    layers

    Attributes:
        hidden_size (int): the size of the second layer

        input_weights (Parameter[torch.tensor, input_dim x hidden_size])
        hidden_biases (Parameter[torch.tensor, hidden_size])
        output_weights (Parameter[torch.tensor, hidden_size x output_dim]
        output_biases (Parameter[torch.tensor, output_dim])

        nonlinearity (torch.tanh): occurs after weights + biases
    """

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int,
                 nonlinearity: typing.Callable,
                 input_weights: torch.tensor, hidden_biases: torch.tensor,
                 output_weights: torch.tensor, output_biases: torch.tensor):
        super().__init__(input_dim, output_dim, 2)

        if not isinstance(hidden_size, int):
            raise ValueError(f'expected hidden_size is int, got {hidden_size} (type={type(hidden_size)})')
        if not callable(nonlinearity):
            raise ValueError(f'expected nonlinearity is callable, got {nonlinearity} (type={type(nonlinearity)})')
        if not torch.is_tensor(input_weights):
            raise ValueError(f'expected input_weights is tensor, got {input_weights} (type={type(input_weights)})')
        if len(input_weights.shape) != 2 or input_weights.shape[0] != input_dim or input_weights.shape[1] != hidden_size:
            raise ValueError(f'expected input_weights is tensor (input_dim, hidden_size) = ({input_dim}, {hidden_size}), got ({input_weights.shape})')
        if not torch.is_tensor(hidden_biases):
            raise ValueError(f'expected hidden_biases is tensor, got {hidden_biases} (type={type(hidden_biases)})')
        if len(hidden_biases.shape) != 1 or hidden_biases.shape[0] != hidden_size:
            raise ValueError(f'expected hidden_biases is tensor (hidden_size) = ({hidden_size}), got {hidden_biases.shape}')
        if not torch.is_tensor(output_weights):
            raise ValueError(f'expected output_weights is tensor, got {output_weights} (type={type(output_weights)})')
        if len(output_weights.shape) != 2 or output_weights.shape[0] != hidden_size or output_weights.shape[1] != output_dim:
            raise ValueError(f'expected output_weights is tensor (hidden_size, output_dim) = ({hidden_size}, {output_dim}), got {output_weights.shape}')
        if not torch.is_tensor(output_biases):
            raise ValueError(f'expected output_biases is tensor, got {output_biases} (type={type(output_biases)})')
        if len(output_biases.shape) != 1 or output_biases.shape[0] != output_dim:
            raise ValueError(f'expected output_biases is tensor (output_dim) = ({output_dim}), got {output_biases.shape}')


        self.hidden_size = hidden_size

        self.input_weights = torch.nn.Parameter(input_weights, requires_grad=True)
        self.hidden_biases = torch.nn.Parameter(hidden_biases, requires_grad=True)
        self.output_weights = torch.nn.Parameter(output_weights, requires_grad=True)
        self.output_biases = torch.nn.Parameter(output_biases, requires_grad=True)

        self.nonlinearity = nonlinearity

    @classmethod
    def create(cls, input_dim: int, hidden_size: int, output_dim: int, nonlinearity: str,
               input_weights: wi.WeightInitializer, hidden_biases: wi.WeightInitializer,
               output_weights: wi.WeightInitializer, output_biases: wi.WeightInitializer):
        """Creates a feedforward network with the given settings. Any of the weights
        may be specified using weight initializers (serialized or not)

        Args:
            input_dim (int): the input dimensionality
            hidden_size (int): the number of hidden nodes in the network
            output_dim (int): the output dimensionality
            nonlinearity (str): either 'tanh' or 'relu'
            input_weights (wi.WeightInitializer): initializes [input_dim x hidden_size]
            hidden_biases (wi.WeightInitializer): initializes [hidden_size]
            output_weights (wi.WeightInitializer): initializes [hidden_size x output_dim]
            output_biases (wi.WeightInitializer): initializes [output_dim]
        """

        if not isinstance(input_dim, int):
            raise ValueError(f'expected input_dim is int, got {input_dim} (type={type(input_dim)})')
        if input_dim <= 0:
            raise ValueError(f'expected input_dim is positive, got {input_dim}')
        if not isinstance(hidden_size, int):
            raise ValueError(f'expected hidden_size is int, got {hidden_size} (type={type(hidden_size)})')
        if hidden_size <= 0:
            raise ValueError(f'expected hidden_size is positive, got {hidden_size}')
        if not isinstance(output_dim, int):
            raise ValueError(f'expected output_dim is int, got {output_dim} (type={type(output_dim)})')
        if output_dim <= 0:
            raise ValueError(f'expected output_dim is positive, got {output_dim}')
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError(f'expected nonlinearity is \'tanh\' or \'relu\', got {nonlinearity}')

        _input_weights = torch.zeros((input_dim, hidden_size), dtype=torch.double)
        wi.deser_or_noop(input_weights).initialize(_input_weights)
        _hidden_biases = torch.zeros((hidden_size), dtype=torch.double)
        wi.deser_or_noop(hidden_biases).initialize(_hidden_biases)
        _output_weights = torch.zeros((hidden_size, output_dim), dtype=torch.double)
        wi.deser_or_noop(output_weights).initialize(_output_weights)
        _output_biases = torch.zeros((output_dim), dtype=torch.double)
        wi.deser_or_noop(output_biases).initialize(_output_biases)

        _nonlinearity = torch.tanh if nonlinearity == 'tanh' else torch.relu
        return cls(input_dim, hidden_size, output_dim, _nonlinearity,
                   _input_weights, _hidden_biases, _output_weights, _output_biases)

    def forward(self, *args):
        if not args:
            raise ValueError('forward(inp[, acts_cb]) expected at least one argument, got 0')

        inp = args[0]
        if not torch.is_tensor(inp):
            raise ValueError(f'forward(inp[, acts_cb]) expected inp is torch.tensor, got {inp} (type={type(inp)})')

        if len(inp.shape) != 2:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape [batch_size, input_dim], got {inp.shape}')

        if inp.shape[1] != self.input_dim:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape [batch_size, input_dim], got {inp.shape} (input_dim={self.input_dim})')

        batch_size = inp.shape[0]

        activations_callback = None if len(args) == 1 else args[1]
        if activations_callback is not None:
            if not callable(activations_callback):
                raise ValueError(f'forward(inp, acts_cb) expected acts_cb is callable, got {activations_callback}')

        if activations_callback:
            activations_callback(FFHiddenActivations(layer=0, hidden_acts=inp))
        activations = self.nonlinearity(inp @ self.input_weights + self.hidden_biases.expand(batch_size, -1))
        if activations_callback:
            activations_callback(FFHiddenActivations(layer=1, hidden_acts=activations))
        activations = self.nonlinearity(activations @ self.output_weights + self.output_biases.expand(batch_size, -1))
        if activations_callback:
            activations_callback(FFHiddenActivations(layer=2, hidden_acts=activations))
        return activations

class FeedforwardLarge(FeedforwardNetwork):
    """Describes a feedforward network of arbitrary size that uses the standard
    torch layers. Does not assume we use the same nonlinearity between all layers. Very
    similar to torch.nn.Sequential, except supporting the feedforwardnetwork
    properties and attributes we expect and splitting the layers and nonlinearities

    Attributes:
        layers (list[torch.nn.Module]): the underlying layers. After each layer we
            apply the nonlinearity before continuing
        nonlinearities (list[callable]): the nonlinearities between layers.
    """

    def __init__(self, layers: typing.List[torch.nn.Module],
                 nonlinearities: typing.List[typing.Callable]):
        super().__init__(layers[0].in_features, layers[-1].out_features, len(layers))

        if not isinstance(nonlinearities, (tuple, list)):
            raise ValueError(f'expected nonlinearities is list, got {nonlinearities} (type={type(nonlinearities)})')
        for idx, nonlin in enumerate(nonlinearities):
            if not callable(nonlin):
                raise ValueError(f'expected nonlinearities[{idx}] is callable, got {nonlin} (type={type(nonlin)})')
        self.layers = layers
        for idx, layer in enumerate(self.layers):
            self.add_module(str(idx), layer)
        self.nonlinearities = nonlinearities

    @classmethod
    def create(cls, input_dim: int, output_dim: int, nonlinearity: typing.Union[typing.List[str], str],
               weights: wi.WeightInitializer, biases: wi.WeightInitializer,
               layer_sizes: typing.Iterable[int]):
        """Creates a feedforward network of linear layers with the particular layer
        sizes. Uses a single weight initializer for all the weights and biases

        Args:
            input_dim (int): the input dimensionality
            output_dim (int): the output dimensionality
            nonlinearity (str or list[str]): one of 'relu', 'tanh', 'none'. may instead be a list
                of equivalent length to layer_sizes
            weights (wi.WeightInitializer): the initializer for the weights between layesr
            biases (wi.WeightInitializer): the initializer for the biases of each layer
            layer_sizes (typing.Iterable[int]): one int for the number of nodes in each layer.
                a single number is similar to FeedforwardSmall with the special weight
                initialization technique. May be empty for an svm
        """

        if not isinstance(input_dim, int):
            raise ValueError(f'expected input_dim is int, got {input_dim} (type={type(input_dim)})')
        if not isinstance(output_dim, int):
            raise ValueError(f'expected output_dim is int, got {output_dim} (type={type(output_dim)})')
        if isinstance(nonlinearity, str):
            if nonlinearity not in nonlins.LOOKUP:
                raise ValueError(f'expected nonlinearity is in {set(nonlins.LOOKUP)}, got {nonlinearity}')
        elif isinstance(nonlinearity, (tuple, list)):
            for idx, val in enumerate(nonlinearity):
                if val not in nonlins.LOOKUP:
                    raise ValueError(f'expected nonlinearity[{idx}] is in {set(nonlins.LOOKUP)}, got {val} (type={type(val)})')
        else:
            raise ValueError(f'expected nonlinearity is str or list[str], got {nonlinearity} (type={type(nonlinearity)}')

        layers = []
        last_dim = input_dim
        for ind, layer_size in enumerate(layer_sizes):
            if not isinstance(layer_size, int):
                raise ValueError(f'expected layers[{ind}] is int, got {layer_size} (type={type(layer_size)})')

            layer = torch.nn.Linear(last_dim, layer_size).double()
            wi.deser_or_noop(weights).initialize(layer.weight.data)
            wi.deser_or_noop(biases).initialize(layer.bias.data)
            layers.append(layer)
            last_dim = layer_size

        layer = torch.nn.Linear(last_dim, output_dim).double()
        wi.deser_or_noop(weights).initialize(layer.weight.data)
        wi.deser_or_noop(biases).initialize(layer.bias.data)
        layers.append(layer)

        _lookup = nonlins.LOOKUP

        if isinstance(nonlinearity, str):
            _nonlins = [_lookup[nonlinearity] for i in range(len(layers))]
        else:
            _nonlins = [_lookup[nonlin] for nonlin in nonlinearity]
            if len(_nonlins) != len(layers):
                raise ValueError(f'expected len(nonlinearity) = len(layers) but got {len(_nonlins)} nonlinearties and {len(layers)} layers')

        return cls(layers, _nonlins)

    def forward(self, *args):
        if not args:
            raise ValueError('forward(inp[, acts_cb]) expected at least one argument, got 0')

        inp = args[0]
        if not torch.is_tensor(inp):
            raise ValueError(f'forward(inp[, acts_cb]) expected inp is torch.tensor, got {inp} (type={type(inp)})')

        if len(inp.shape) != 2:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape [batch_size, input_dim], got {inp.shape}')

        if inp.shape[1] != self.input_dim:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape [batch_size, input_dim], got {inp.shape} (input_dim={self.input_dim})')

        activations_callback = None if len(args) == 1 else args[1]
        if activations_callback is not None:
            if not callable(activations_callback):
                raise ValueError(f'forward(inp, acts_cb) expected acts_cb is callable, got {activations_callback}')

        if activations_callback:
            activations_callback(FFHiddenActivations(layer=0, hidden_acts=inp))

        activations = inp
        for layer_ind, layer in enumerate(self.layers):
            activations = self.nonlinearities[layer_ind](layer(activations))
            if activations_callback:
                activations_callback(FFHiddenActivations(layer=layer_ind + 1, hidden_acts=activations))

        return activations

class ComplexLayer(typing.NamedTuple):
    """Describes a layer in the FeedforwardComplex network with some additional
    attributes for plotting purposes. We assume that if the layer is not a module
    then in_features is arbitrary and out_features matches in_features

    Attributes:
        style (str): one of 'layer', 'nonlinearity', 'other'
        is_module (bool): True to store under modules, False not to
        invokes_callback (bool): True if the network state should be
            visualized after the action is invoked, False if this is
            an intermediary layer. For example, a Linear module
            + tanh nonlinearity should only invoke the callback after
            the nonlinearity
        action (callable): the thing which will be called with the
            hidden activations and will return the new activations
    """

    style: str
    is_module: bool
    invokes_callback: bool
    action: typing.Callable

class FeedforwardComplex(FeedforwardNetwork):
    """Describes a network which might use different nonlinearities and nonlinear
    layers (such as convolutional layers)

    Attributes:
        layers (list[ComplexLayer]): the layers that are in this network
    """

    def __init__(self, input_dim: int, output_dim: int, layers: typing.List[ComplexLayer]):
        if not isinstance(input_dim, int):
            raise ValueError(f'expected input dim is int, got {input_dim} (type={type(input_dim)})')
        if not isinstance(output_dim, int):
            raise ValueError(f'expected output_dim is int, got {output_dim} (type={type(output_dim)})')
        if not isinstance(layers, (list, tuple)):
            raise ValueError(f'expected layers is list-like, got {layers} (type={type(layers)})')

        num_layers = 0
        for lyr in layers:
            if lyr.invokes_callback:
                num_layers += 1

        super().__init__(input_dim, output_dim, num_layers)
        self.layers = layers

        for idx, lyr in enumerate(layers):
            if lyr.is_module:
                self.add_module(str(idx), lyr.action)

    def forward(self, *args):
        if not args:
            raise ValueError('forward(inp[, acts_cb]) expected at least one argument, got 0')

        inp = args[0]
        inp = inp.float() # todo
        if not torch.is_tensor(inp):
            raise ValueError(f'forward(inp[, acts_cb]) expected inp is torch.tensor, got {inp} (type={type(inp)})')

        if len(inp.shape) != 2:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape [batch_size, input_dim], got {inp.shape}')

        if inp.shape[1] != self.input_dim:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape [batch_size, input_dim], got {inp.shape} (input_dim={self.input_dim})')

        activations_callback = None if len(args) == 1 else args[1]
        if activations_callback is not None:
            if not callable(activations_callback):
                raise ValueError(f'forward(inp, acts_cb) expected acts_cb is callable, got {activations_callback}')

        activations = inp
        #print(f'input: shape={inp.shape}, dtype={inp.dtype}')
        if activations_callback:
            activations_callback(FFHiddenActivations(layer=0, hidden_acts=inp.double()))

        layer_ind = 1
        for idx, lyr in enumerate(self.layers):
            activations = lyr.action(activations)
            print(f'after layer {idx} (style={lyr.style}), shape={activations.shape}, dtype={activations.dtype}')
            if lyr.invokes_callback and activations_callback:
                cb_activations = activations.double()
                if len(activations.shape) != 2:
                    cb_activations = activations.reshape(
                        activations.shape[0],
                        reduce(operator.mul, activations.shape[1:]))
                    #print(f'after reshaping for callback: {cb_activations.shape}')
                activations_callback(FFHiddenActivations(layer=layer_ind, hidden_acts=cb_activations))
                layer_ind += 1

        return activations


class FFTeacher(NetworkTeacher):
    """A simple network teacher
    """
    def teach_many(self, network: Network, optimizer: torch.optim.Optimizer, criterion: typing.Any,
                   points: torch.tensor, labels: torch.tensor):
        network.train()
        with torch.set_grad_enabled(True):
            network.zero_grad()
            optimizer.zero_grad()
            result = network(points)
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()
        return loss.item()

    def classify_many(self, network: Network, points: torch.tensor, out: torch.tensor):
        network.eval()
        with torch.no_grad():
            result = network(points)
        out.copy_(result)
