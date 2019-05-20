"""Contains a sequence to sequence charceter model based on an encoder->decoder framework.
Chosen from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"""

import typing
import torch
import torch.nn as nn

import shared.typeutils as tus
import shared.nonlinearities as nonlins
from shared.models.generic import Network
from shared.teachers import SeqSeqTeacher, Sequence

class EncoderRNN(Network):
    """Describes the encoding rnn. This is a sequence to vector network. It first goes through
    a linear network to go from the input dimensionality to GRU dimensionality. Then it goes
    through the GRU to get an output and hidden state. Those two are stacked and sent through
    a linear network.

    Attributes:
        hidden_size (int): the size of the hidden space

        input_dim (int): the number of floats presented during each timestep
        output_dim (int): the number of floats returned at each timestep
        num_layers (int): the number of hidden layers

        in_interpreter (torch.nn.Linear): a linear layer that scales from the input dimension
            to the hidden dimension for the input to the encoder
        in_nonlinearity (tanh): the nonlinearity for the input interpreter

        gru (torch.nn.GRU): the GRU that is used for encoding

        out_interpreter (torch.nn.Linear): a simple linear layer that converts from
            the output of the GRU to the actual output format we want
        out_nonlinearity (tanh): the nonlinearity for the output interpreter
    """

    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1):
        super().__init__(input_dim, output_dim)
        tus.check(input_dim=(input_dim, int), hidden_size=(hidden_size, int),
                  output_dim=(output_dim, int), num_layers=(num_layers, int))

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.in_interpreter = nn.Linear(input_dim, hidden_size).double()
        self.in_nonlinearity = torch.tanh
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True).double()
        self.out_interpreter = nn.Linear(hidden_size * (num_layers + 1), output_dim).double()
        self.out_nonlinearity = torch.tanh

    def transfer_up(self, new_hidden_size, new_output_dim) -> 'EncoderRNN':
        """Returns a new encoder rnn with the specified hidden size and this network embedded
        into it"""
        tus.check(new_hidden_size=(new_hidden_size, int), new_output_dim=(new_output_dim, int))
        if new_hidden_size < self.hidden_size:
            raise ValueError(f'cannot transfer to hidden size {new_hidden_size} from {self.hidden_size}')
        if new_output_dim < self.output_dim:
            raise ValueError(f'cannot transfer to output dim {new_output_dim} from {self.output_dim}')

        copy = EncoderRNN(self.input_dim, new_hidden_size, new_output_dim, self.num_layers)

        copy.in_interpreter.weight.data[:self.hidden_size, :] = self.in_interpreter.weight.data
        copy.in_interpreter.bias.data[:self.hidden_size] = self.in_interpreter.bias.data
        copy.gru.weight_ih_l0.data[:3*self.hidden_size, :self.hidden_size] = self.gru.weight_ih_l0.data
        copy.gru.bias_ih_l0.data[:3*self.hidden_size] = self.gru.bias_ih_l0.data

        new_out_w = copy.out_interpreter.weight.data
        old_out_w = self.out_interpreter.weight.data

        new_out_w = new_out_w.view(new_output_dim, self.num_layers + 1, new_hidden_size)
        old_out_w = old_out_w.view(self.output_dim, self.num_layers + 1, self.hidden_size)

        new_out_w[:self.output_dim, :, :self.hidden_size] = old_out_w

        copy.out_interpreter.bias.data[:self.output_dim] = self.out_interpreter.bias.data
        return copy

    def forward(self, inp: torch.tensor) -> torch.tensor: # pylint: disable=arguments-differ, line-too-long
        """Presents the given string to the network and returns the context state

        Arguments:
            inp (torch.tensor [batch size, sequence length, input size]): tensor containing the
                input features. May send multiple batches of the same sequence length at once.
                The batch is indicated with the first index.  The second index corresponds to the
                timestep of the sequence, and the last index tells us which feature

        Returns:
            context_vector (torch.tensor[batch_size, output_size])
        """
        tus.check_tensors(inp=(inp, (('batch_size', None), ('sequence length', None),
                                     ('input size', self.input_dim)), torch.double))
        interpinp = self.in_nonlinearity(
            self.in_interpreter(inp.reshape(inp.shape[0] * inp.shape[1], inp.shape[2]))
        ).reshape(inp.shape[0], inp.shape[1], self.hidden_size)
        out1, out2 = self.gru(interpinp)
        out2 = out2.transpose(0, 1) # pytorch bug I'm fairly sure
        tus.check_tensors(out1=(out1, (('batch_size', inp.shape[0]), ('sequence length', inp.shape[1]),
                                       ('hidden_size', self.hidden_size)), torch.double),
                          out2=(out2, (('batch_size', inp.shape[0]), ('num layers', self.num_layers),
                                       ('hidden_size', self.hidden_size)), torch.double))
        out1_last = out1[:, -1, :].reshape(out1.shape[0], self.hidden_size)
        out2_reshaped = out2.reshape(out2.shape[0], self.num_layers * self.hidden_size)
        stacked = torch.cat((out1_last, out2_reshaped), dim=1)
        out = self.out_nonlinearity(self.out_interpreter(stacked))
        tus.check_tensors(out=(out, (('batch_size', inp.shape[0]),
                                     ('output_size', self.output_dim)), torch.double))
        return out

class DecoderRNN(Network):
    """Describes the decoding rnn. This takes the context vector, sends it through a linear
    network to get it to the correct size, then runs it through a GRU to get an output which
    is passed through a linear network and returned. This process is repeated until the caller
    tells us to stop.

    Attributes:
        input_dim (int): the size of the context vector
        hidden_size (int): the number of neurons in the GRU
        output_dim (int): the size of each output from the GRU
        num_layers (int): the depth of the GRU

        input_interpreter (nn.Linear): converts from the context vector to the GRU input
        input_nonlinearity (tanh): nonlinearity applied after input interpreter
        gru (GRU): the gru layer
        output_interpreter (nn.Linear): converts from the GRU output to the output state
        output_nonlinearity (isrlu): nonlinearity applied before returning
    """

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers=1):
        super().__init__(input_dim, output_dim)
        tus.check(input_dim=(input_dim, int), hidden_size=(hidden_size, int),
                  output_dim=(output_dim, int), num_layers=(num_layers, int))
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.input_interpreter = nn.Linear(input_dim, hidden_size).double()
        self.input_nonlinearity = torch.tanh
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True).double()
        self.output_interpreter = nn.Linear(hidden_size * (num_layers + 1), output_dim).double()
        self.output_nonlinearity = nonlins.extended_lookup('isrlu')

    def transfer_up(self, new_input_dim, new_hidden_size) -> 'DecoderRNN':
        """Creates a new decoder rnn with the specified input and hidden size with this decoder
        embedded in it
        """
        tus.check(new_input_dim=(new_input_dim, int), new_hidden_size=(new_hidden_size, int))
        if new_input_dim < self.input_dim:
            raise ValueError(f'cannot transfer to input dim {new_input_dim} from input dim {self.input_dim}')
        if new_hidden_size < self.hidden_size:
            raise ValueError(f'cannot transfer to hidden size {new_hidden_size} from hidden size {self.hidden_size}')

        copy = DecoderRNN(new_input_dim, new_hidden_size, self.output_dim, self.num_layers)

        copy.input_interpreter.weight.data[:self.hidden_size, :self.input_dim] = self.input_interpreter.weight.data
        copy.input_interpreter.bias.data[:self.hidden_size] = self.input_interpreter.bias.data
        copy.gru.weight_ih_l0.data[:3*self.hidden_size, :self.hidden_size] = self.gru.weight_ih_l0.data
        copy.gru.bias_ih_l0.data[:3*self.hidden_size] = self.gru.bias_ih_l0.data

        new_out_w = copy.output_interpreter.weight.data
        old_out_w = self.output_interpreter.weight.data

        new_out_w = new_out_w.view(self.output_dim, self.num_layers + 1, new_hidden_size)
        old_out_w = old_out_w.view(self.output_dim, self.num_layers + 1, self.hidden_size)

        new_out_w[:, :, :self.hidden_size] = old_out_w

        copy.output_interpreter.bias.data[:self.output_dim] = self.output_interpreter.bias.data
        return copy


    def forward(self, inp: torch.tensor, callback: typing.Callable = None) -> torch.tensor: # pylint: disable=arguments-differ
        """Goes from the input vector to a sequence. The sequence is returned by invoking the
        callback at each step, which must return True if the network needs to continue or False
        if the network is finished. This operation cannot be batched.

        If you need to do teacher-forcing you can have the callback return True and then reinvoke

        Arguments:
            inp (tensor [input_dim]): the context vector to forward through the network
            callback (function(output (tensor [output_dim])) -> bool, optional): accepts the step
                of the network and returns True to continue generating a sequence and False to stop.
                If None, acts like lambda out: True.

        Returns:
            out (tensor [output_dim]): the last output of the network
        """
        tus.check_tensors(inp=(inp, [('input_dim', self.input_dim)], torch.double))
        inp_interp = self.context_to_hidden(inp)

        hidden, state = self.hidden_through_gru(inp_interp, None)
        if callback is None or not callback(self.hidden_to_output(hidden, state)):
            return hidden, state

        while True:
            hidden, state = self.hidden_through_gru(hidden, state)
            if not callback(self.hidden_to_output(hidden, state)):
                return hidden, state

    def context_to_hidden(self, context: torch.tensor) -> torch.tensor:
        """Converts the context tensor to the hidden input"""
        tus.check_tensors(context=(context, [('context', self.input_dim)], torch.double))
        return self.input_nonlinearity(self.input_interpreter(context))

    def hidden_through_gru(self, hidden: torch.tensor, state: torch.tensor) -> typing.Tuple[torch.tensor, torch.tensor]:
        """Sends the hidden tensor through the GRU and returns the new hidden tensor and the
        new hidden state"""
        if state is None:
            state = torch.zeros((self.num_layers, self.hidden_size,), dtype=torch.double)

        tus.check_tensors(
            hidden=(hidden, [('hidden_size', self.hidden_size)], torch.double),
            state=(state, [('num_layers', self.num_layers), ('hidden_size', self.hidden_size)], torch.double)
        )
        out1, out2 = self.gru(hidden.reshape(1, 1, -1), state.reshape(self.num_layers, 1, self.hidden_size))
        out2 = out2.transpose(0, 1)
        tus.check_tensors(
            out1=(out1, [('batch_size', 1), ('sequence_length', 1), ('hidden_size', self.hidden_size)], torch.double),          # pylint: disable=line-too-long
            out2=(out2, [('batch_size', 1), ('num layers', self.num_layers), ('hidden_size', self.hidden_size)], torch.double)  # pylint: disable=line-too-long
        )
        return out1.reshape(self.hidden_size), out2.reshape(self.num_layers, self.hidden_size)

    def hidden_to_output(self, hidden: torch.tensor, state: torch.tensor) -> torch.tensor:
        """Converts the hidden vector and state to the output of the network"""
        tus.check_tensors(
            hidden=(hidden, [('hidden_size', self.hidden_size)], torch.double),
            state=(state, [('num_layers', self.num_layers), ('hidden_size', self.hidden_size)], torch.double)
        )
        stacked = torch.cat((hidden, state.reshape(self.num_layers * self.hidden_size)), dim=0)
        return self.output_nonlinearity(self.output_interpreter(stacked))

class EncoderDecoder(Network):
    """Describes an encoder-decoder sequence-sequence RNN. It is comprised, as the name suggests,
    of an encoder (seq->vec) and decoder (vec->seq)

    Attributes:
        input_dim (int): the size of the input steps (arbitrary number of steps)
        encoding_dim (int): the size of the encoder hidden representation
        context_dim (int): the size of the context vector which the encoder targets
        decoding_dim (int): the size of the decoder hidden representation
        output_dim (int): the size of the output steps (arbitrary number of steps)
        encoding_layers (int): the number of encoding GRU layers
        decoding_layers (int): the number of decoding GRU layers

        encoder (EncoderRNN): the encoder
        decoder (DecoderRNN): the decoder
    """

    def __init__(self, input_dim: int, encoding_dim: int, context_dim: int, decoding_dim: int,
                 output_dim: int, encoding_layers=1, decoding_layers=1):
        super().__init__(input_dim, output_dim)

        tus.check(input_dim=(input_dim, int), encoding_dim=(encoding_dim, int),
                  context_dim=(context_dim, int),
                  decoding_dim=(decoding_dim, int), output_dim=(output_dim, int),
                  encoding_layers=(encoding_layers, int), decoding_layers=(decoding_layers, int))

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.context_dim = context_dim
        self.decoding_dim = decoding_dim
        self.output_dim = output_dim

        self.encoder = EncoderRNN(input_dim, encoding_dim, context_dim,
                                  num_layers=encoding_layers)
        self.decoder = DecoderRNN(context_dim, decoding_dim, output_dim,
                                  num_layers=decoding_layers)

    def transfer_up(self, new_encoding_dim, new_context_dim, new_decoding_dim):
        """Returns a new encoder decoder network with the specified encoding, context, and decoding dim
        at least as large as this one. Has this network embedded into it."""
        tus.check(new_encoding_dim=(new_encoding_dim, int), new_context_dim=(new_context_dim, int),
                  new_decoding_dim=(new_decoding_dim, int))

        copy = EncoderDecoder(self.input_dim, new_encoding_dim, new_context_dim, new_decoding_dim,
                              self.output_dim, 1, 1)
        copy.encoder = self.encoder.transfer_up(new_encoding_dim, new_context_dim)
        copy.decoder = self.decoder.transfer_up(new_context_dim, new_decoding_dim)
        return copy

    def forward(self, inp: torch.tensor, callback: typing.Callable) -> None: # pylint: disable=arguments-differ
        """Forwards through the network. This is not sufficient information for training really,
        which should probably understand the encoding / decoding dichotomy. However, it does make
        the forward pass pretty easy.

        Attributes:
            inp (torch.tensor [input_sequence_length, input_dim]): the sequence to present to the
                network.
            callback (function(tensor[output_dim]) -> bool): the callback is presented with each
                output element of the sequence until it returns False
        """

        context = self.encoder(inp.unsqueeze(dim=0))
        self.decoder(context.squeeze(), callback)

class EDReader:
    """Callable class that stores the output as a sequence until we reach a certain
    length or another callable tells us to stop

    Attributes:
        output (list[tuple[float]]): the outputs that we've seen so far
        stop_failer (callable): returns True to continue, False to stop
        max_out_len (int): the maximum number of results before we just call it
    """

    def __init__(self, stop_failer: typing.Callable, max_out_len: int) -> None:
        tus.check(max_out_len=(max_out_len, int))
        tus.check_callable(stop_failer=stop_failer)
        self.stop_failer = stop_failer
        self.max_out_len = max_out_len
        self.output = []

    def __call__(self, output: torch.tensor) -> bool:
        self.output.append(tuple(a.item() for a in output))
        return len(self.output) < self.max_out_len and self.stop_failer(output)

class EncoderDecoderTeacher(SeqSeqTeacher):
    """Describes a teacher for an encoder/decoder network. Uses a random percent
    of teacher forcing.

    Attributes:
        stop_failer (callable): accepts a tensor of shape [output_dim] and returns False
            stop and True to continue.
        max_out_len (int): the maximum number of outputs from the network before we just
            assume its not going to tell us to stop
    """

    def __init__(self, stop_failer: typing.Callable, max_out_len: int):
        tus.check(max_out_len=(max_out_len, int))
        tus.check_callable(stop_failer=stop_failer)

        self.max_out_len = max_out_len
        self.stop_failer = stop_failer

    def teach_single(self, network: EncoderDecoder, optimizers: typing.List[torch.optim.Optimizer],
                     criterion: typing.Any, inp: Sequence, out: Sequence) -> float:
        """Teaches the network with a single input / output sequence"""
        network.train()

        with torch.set_grad_enabled(True):
            network.zero_grad()
            for optim in optimizers:
                optim.zero_grad()
            encoder_input = inp.raw if torch.is_tensor(inp.raw) else torch.tensor(inp.raw, dtype=torch.double) # pylint: disable=not-callable, line-too-long
            context = network.encoder(encoder_input.unsqueeze(dim=0)).squeeze()

            loss = 0
            cur_hidden, cur_state = network.decoder.context_to_hidden(context), None
            for exp in out.raw:
                exp_torch = exp if torch.is_tensor(exp) else torch.tensor(exp, dtype=torch.double) # pylint: disable=not-callable, line-too-long
                cur_hidden, cur_state = network.decoder.hidden_through_gru(cur_hidden, cur_state)
                act = network.decoder.hidden_to_output(cur_hidden, cur_state)

                loss += criterion(act, exp_torch)

            loss.backward()
            for optim in optimizers:
                optim.step()

        return loss.item()

    def teach_many(self, network: EncoderDecoder, optimizers: typing.List[torch.optim.Optimizer],
                   criterion: typing.Any, inputs: typing.List[Sequence],
                   outputs: typing.List[Sequence]) -> float:
        all_losses = torch.zeros(len(inputs), dtype=torch.double)
        for ind, inp in enumerate(inputs):
            out = outputs[ind]
            all_losses[ind] = self.teach_single(network, optimizers, criterion, inp, out)
        return all_losses.mean().item()

    def classify_many(self, network: EncoderDecoder,
                      inputs: typing.List[Sequence]) -> typing.List[Sequence]:
        result = []
        for inp in inputs:
            inp_tensor = inp.raw if torch.is_tensor(inp.raw) else torch.tensor(inp.raw, dtype=torch.double) # pylint: disable=not-callable, line-too-long
            reader = EDReader(self.stop_failer, self.max_out_len)
            network(inp_tensor, reader)
            result.append(Sequence(raw=reader.output))
        return result
