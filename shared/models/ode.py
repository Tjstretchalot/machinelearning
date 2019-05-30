"""An ordinary differential equations network, which is basically a
feedforward network with different arguments"""
import torch
from torchdiffeq import odeint
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations

class ODEfunc(torch.nn.Module):
    """Describes the function that can be used by the ODE to get the new hidden
    state of the network. This name comes directly from the source

    Attributes:
        real_net (FeedForwardNetwork) the real network
    """

    def __init__(self, real_net: FeedforwardNetwork) -> None:
        super().__init__()
        if not isinstance(real_net, FeedforwardNetwork):
            raise ValueError(f'expected real_net is FeedforwardNetwork, got {real_net} (type={type(real_net)})')

        self.real_net = real_net

    def forward(self, evaltime: torch.tensor, curhid: torch.tensor) -> torch.tensor: #pylint: disable=arguments-differ
        """This function computes the derivative of the hidden state at
        a given time "evaltime", given the current statei s "curhid" and we have
        our current set of parameters (given by real_net)"""
        return self.real_net(curhid)

class ODEBlock(FeedforwardNetwork):
    """This wraps a network that is meant to be evaluated with an ODE such that it
    appears as a 2-layer feedforward network, where the middle layer is actually the
    recurrent network. So you would train with ODEfunc but then to use the feed
    forward metrics you use an ODEBlock

    Attributes:
        wrapped_net (ODEfunc): the wrapped network that is being trained
        integration_time (torch.tensor): the times the network is given
    """

    def __init__(self, real_net: FeedforwardNetwork, integration_time=None):
        super().__init__(real_net.input_dim, real_net.output_dim, 1)
        self.wrapped_net = ODEfunc(real_net)
        if integration_time is None:
            integration_time = torch.tensor([0, 1], dtype=torch.double)
        self.integration_time = integration_time

    def forward(self, *args):
        if not args:
            raise ValueError('forward(inp[, acts_cb]) expected at least one argument, got 0')

        inp = args[0]
        if not torch.is_tensor(inp):
            raise ValueError(f'forward(inp[, acts_cb]) expected inp is torch.tensor, '
                             + f'got {inp} (type={type(inp)})')

        if len(inp.shape) != 2:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape '
                             + f'[batch_size, input_dim], got {inp.shape}')

        if inp.shape[1] != self.input_dim:
            raise ValueError(f'forward(inp[, acts_cb]) expected inp has shape '
                             + f'[batch_size, input_dim], got {inp.shape} '
                             + f'(input_dim={self.input_dim})')

        activations_callback = None if len(args) == 1 else args[1]
        if activations_callback is not None:
            if not callable(activations_callback):
                raise ValueError(f'forward(inp, acts_cb) expected acts_cb is callable, got {activations_callback}')

        activations = inp
        if activations_callback:
            activations_callback(FFHiddenActivations(layer=0, hidden_acts=activations.double()))

        activations = odeint(self.wrapped_net, activations,
                             self.integration_time.type_as(activations))[1]

        if activations_callback:
            activations_callback(FFHiddenActivations(layer=1, hidden_acts=activations.double()))

        return activations
