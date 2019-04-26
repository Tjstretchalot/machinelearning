"""A collection of utility functions for measures"""

import typing
import torch
import numpy as np
from shared.models.generic import Network
from shared.models.ff import FeedforwardNetwork, FFHiddenActivations
from shared.models.rnn import NaturalRNN, RNNHiddenActivations
from shared.pwl import PointWithLabelProducer
from shared.trainer import GenericTrainingContext
import shared.npmp as npmp
import os

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
    if sample_labels.dtype == torch.uint8:
        raise ValueError(f'uint8 sample_labels is prone to issues with masking, convert to int or long')
    if sample_labels.dtype not in (torch.uint8, torch.int, torch.long):
        raise ValueError(f'expected sample_labels is int-like but has dtype={sample_labels.dtype}')
    if sample_points.shape[0] != sample_labels.shape[0]:
        raise ValueError(f'expected sample_points has shape [num_pts, input_dim] and sample_labels has shape [num_pts] but sample_points.shape={sample_points.shape} and sample_labels.shape={sample_labels.shape} (mismatch on dim 0)')

def verify_points_and_labels_np(sample_points: np.ndarray, sample_labels: np.ndarray):
    """Verifies that the sample points and sample labels go together and make sense
    and are numpy-style

    Args:
        sample_points (np.ndarray): the sample points (as numpy)
        sample_labels (np.ndarray): the sample labels (as numpy)
    """
    verify_points_and_labels(torch.from_numpy(sample_points), torch.from_numpy(sample_labels))

class NetworkHiddenActivations:
    """Describes the result from get_hidacts style functions. Describes the hidden
    activations of a network through layers or through time, depending on if the
    network is feed forward or not.

    Attributes:
        netstyle (str): one of 'feedforward' or 'recurrent'
        sample_points (tensor[num_pts, input_dim], float-like):
            the points which were used to sample. This is a numpy array after a call
            to numpy()
        sample_labels (tensor[num_pts], int-like): the labels of the points. This is a
            numpy array after a call to numpy()

        hid_acts (list[num_layers of tensor[num_pts, layer_size], float-like):
            the activations of the network as you go through time. accessible through
            time or layers, depending on the netstyle. These are numpy arrays after
            a call to numpy()
    """

    def __init__(self, netstyle: str, sample_points: torch.tensor, sample_labels: torch.tensor,
                 hid_acts: typing.List[torch.tensor]) -> None:
        self.netstyle = netstyle
        self.sample_points = sample_points
        self.sample_labels = sample_labels
        self.hid_acts = hid_acts

        self.verify()

    def numpy(self):
        """Changes this to numpy style from torch style.
        """
        if self.is_numpy:
            return self
        self.sample_points = self.sample_points.numpy()
        self.sample_labels = self.sample_labels.numpy()
        for i in range(len(self.hid_acts)):
            self.hid_acts[i] = self.hid_acts[i].numpy()
        return self

    def torch(self):
        """Changes this to torch style from numpy style.
        """
        if not self.is_numpy:
            return self
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

    @property
    def label(self):
        """Gives a reasonable name for the x-axis based on the netstyle
        """

        if self.netstyle == 'feedforward':
            return 'Layers'
        elif self.netstyle == 'recurrent':
            return 'Time'
        raise ValueError(f'unknown netstyle {self.netstyle}')

    @property
    def is_numpy(self):
        """Determines if this is currently using numpy arrays (True) or torch
        tensors (False)
        """
        return not torch.is_tensor(self.sample_points)

    def verify(self):
        """Verifies that this is a valid set of activations, i.e., there are no obvious
        logical inconsistencies
        """
        if self.netstyle not in ('feedforward', 'recurrent'):
            raise ValueError(f'expected netstyle is \'feedforward\' or \'recurrent\', got {self.netstyle} (type={type(self.netstyle)})')

        is_numpy = self.is_numpy
        if is_numpy:
            verify_points_and_labels_np(self.sample_points, self.sample_labels)
        else:
            verify_points_and_labels(self.sample_points, self.sample_labels)
        if not isinstance(self.hid_acts, list):
            raise ValueError(f'expected hid_acts is list[tensor[num_pts, layer_size]] but is {self.hid_acts} (type={type(self.hid_acts)})')
        for idx, lyr in enumerate(self.hid_acts):
            if (is_numpy and not isinstance(lyr, np.ndarray)) or (not is_numpy and not torch.is_tensor(lyr)):
                raise ValueError(f'expected hid_acts[{idx}] is tensor[num_pts, layer_size] but is not tensor: {lyr} (type={type(lyr)}) is_numpy={is_numpy}')
            if len(lyr.shape) != 2:
                raise ValueError(f'expected hid_acts[{idx}].shape is [num_pts, layer_size] but is {lyr.shape}')
            if lyr.shape[0] != self.sample_points.shape[0]:
                raise ValueError(f'expected hid_acts[{idx}].shape is [num_pts={self.sample_points.shape[0]}, layer_size] but is {lyr.shape}')
            if ((not is_numpy and lyr.dtype not in (torch.float, torch.double))
                    or (is_numpy and lyr.dtype not in (np.float, np.float32, np.float64))):
                raise ValueError(f'expected hid_acts[{idx}] is float-like but dtype={lyr.dtype}')




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
        num_points = 50*network.output_dim

    if not isinstance(network, FeedforwardNetwork):
        raise ValueError(f'expected network is FeedforwardNetwork, got {network} (type={type(network)})')
    if not isinstance(pwl, PointWithLabelProducer):
        raise ValueError(f'expected pwl is PointWithLabelProducer, got {pwl} (type={type(pwl)})')
    if not isinstance(num_points, int):
        raise ValueError(f'expected num_points is int, got {num_points} (type={type(num_points)})')

    num_points = min(num_points, pwl.epoch_size)

    sample_points = torch.zeros((num_points, pwl.input_dim), dtype=torch.double)
    sample_labels = torch.zeros(num_points, dtype=torch.int)

    pwl.mark()
    pwl.fill(sample_points, sample_labels)
    pwl.reset()
    return get_hidacts_ff_with_sample(network, sample_points, sample_labels)

def get_hidacts_rnn_with_sample(network: NaturalRNN, sample_points: torch.tensor,
                                sample_labels: torch.tensor) -> NetworkHiddenActivations:
    """Gets the hidden activations for a recurrent network when running the given sample
    points through it, at attaches the given sample labels to the result.

    Args:
        network (NaturalRNN): the network to forward through
        sample_points (torch.tensor): the points to forward through
        sample_labels (torch.tensor): the labels of the points which will be forwarded

    Returns:
        NetworkHiddenActivations: the activations of the network
    """
    if not isinstance(network, NaturalRNN):
        raise ValueError(f'expected network is NaturalRNN, got {network} (type={type(network)})')
    verify_points_and_labels(sample_points, sample_labels)

    hid_acts = []

    def on_hidacts(acts_info: RNNHiddenActivations):
        hid_acts.append(acts_info.hidden_acts.detach())

    network(sample_points, sample_labels, on_hidacts)
    return NetworkHiddenActivations('recurrent', sample_points, sample_labels, hid_acts)

def get_hidacts_rnn(network: NaturalRNN, pwl: PointWithLabelProducer,
                    num_points: typing.Optional[int] = None) -> NetworkHiddenActivations:
    """Gets the hidden activations for the given recurrent network, acquiring at most
    num_points from the producer.

    Args:
        network (NaturalRNN): the network to forward through
        pwl (PointWithLabelProducer): the producer for points to run through
        num_points (typing.Optional[int], optional): Defaults to None. the maximum number
            of points to run through the network. Clipped to pwl.epoch_size

    Returns:
        NetworkHiddenActivations: the internal activations of the network
    """
    if num_points is None:
        num_points = 50*network.output_dim

    if not isinstance(network, NaturalRNN):
        raise ValueError(f'expected network is FeedforwardNetwork, got {network} (type={type(network)})')
    if not isinstance(pwl, PointWithLabelProducer):
        raise ValueError(f'expected pwl is PointWithLabelProducer, got {pwl} (type={type(pwl)})')
    if not isinstance(num_points, int):
        raise ValueError(f'expected num_points is int, got {num_points} (type={type(num_points)})')

    sample_points = torch.zeros((num_points, pwl.input_dim), dtype=torch.double)
    sample_labels = torch.zeros(num_points, dtype=torch.int)

    pwl.mark()
    pwl.fill(sample_points, sample_labels)
    pwl.reset()
    return get_hidacts_rnn_with_sample(network, sample_points, sample_labels)

def get_hidacts_with_sample(network: Network, sample_points: torch.tensor,
                            sample_labels: torch.tensor) -> NetworkHiddenActivations:
    """Runs the given tensors through the network and returns the networks hidden activations.

    Args:
        network (Network): the network
        sample_points (torch.tensor): the points to run through the network
        sample_labels (torch.tensor): the labels for the points
    """

    if isinstance(network, FeedforwardNetwork):
        return get_hidacts_ff_with_sample(network, sample_points, sample_labels)
    if isinstance(network, NaturalRNN):
        return get_hidacts_rnn_with_sample(network, sample_points, sample_labels)

    raise ValueError(f'unknown network type {network} (type={type(network)})')


def process_outfile(outfile: str, exist_ok: bool) -> typing.Tuple[str, str]:
    """Checks if the given outfile and exist_ok combination is valid. Returns
    the outfile and the outfile_wo_ext, one of which will be the outfile passed.

    Args:
        outfile (str): the file you were told to output to
        exist_ok (bool): True if the file should be overwritten, False otherwise

    Returns:
        outfile (str): the filepath with extension '.zip' to save to
        outfile_wo_ext (str): the filepath without an extension
    """
    if not isinstance(outfile, str):
        raise ValueError(f'expected outfile is str, got {outfile} (type={type(outfile)})')
    if not isinstance(exist_ok, bool):
        raise ValueError(f'expected exist_ok is bool, got {exist_ok} (type={type(exist_ok)})')

    outfile_wo_ext, ext = os.path.splitext(outfile)
    if ext not in ('', '.zip'):
        raise ValueError(f'expected outfile is .zip (extension may be excluded), got {outfile} (ext={ext})')

    if os.path.exists(outfile_wo_ext):
        raise FileExistsError(f'in order to save to {outfile} need {outfile_wo_ext} as working space')

    outfile = outfile_wo_ext + '.zip'
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(f'cannot save to {outfile} (already exists) (set exist_ok=True to overwrite)')

    return outfile, outfile_wo_ext

def digest_std(measure: typing.Callable, plot: typing.Callable):
    """This should be used in association with during_training_std to setup a target
    for a particular module. This creates the target function expected by the
    during_training_std and works for the training.

    The result should be called 'digest'

    Args:
        measure (typing.Callable): the measure function which accepts
            NetworkHiddenActivations in numpy style as its first argument, followed by
            arbitrary kwargs

        plot (typing.Callable): the plotter function which accepts the result form measure
            as its first argument, followed by the outfile, followed by arbitrary kwargs

    Returns:
        target (typing.Callable): the target for during_training_std
    """

    def target(sample_points: np.ndarray, sample_labels: np.ndarray,
               *hid_acts: np.ndarray, outfile: str = None, netstyle: str = None,
               measure_kwargs: dict = None, plot_kwargs: dict = None):
        measure_kwargs = measure_kwargs if measure_kwargs is not None else dict()
        plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()

        hacts = NetworkHiddenActivations(netstyle, sample_points, sample_labels, list(hid_acts))
        traj = measure(hacts, **measure_kwargs)
        plot(traj, outfile, **plot_kwargs)

    return target

def during_training_std(identifier: str, measure: typing.Callable, plot: typing.Callable,
                        target_module: str,
                        num_points: typing.Optional[typing.Union[typing.Callable, int]] = None):
    """Returns a function that is meant to be called as follows:

    during_training(savepath: str, train: bool, digestor: typing.Optional[npmp.NPDigestor] = None,
                    measure_args: dict = None, plot_args: dict = None)

    Where measure_args and plot_args are passed to measure and plot respectively, if they
    are specified. The result of during_training_std is the during_training which returns
    the on_step function which can be passed to something like an EpochCaller

    This function automatically detects FeedforwardNetwork vs NaturalRNNs


    Args:
        identifier (str): the identifier for the action that this refers to, i.e,
            SVM
        measure (typing.Callable): the measure function which accepts
            (hacts: NetworkHiddenActivations, **kwargs) in numpy-style
            and returns something which is typically called a Trajectory
        plot (typing.Callable): the plot function which accepts
            (traj, outfile, exist_ok=False, xlabel=None, **kwargs) where traj is the result
            from measure_np. the xlabel comes from the trajectory
        num_points (int or callable or None): either an int for a specific target number of
            points, or a callable which accepts the GenericTrainingContext and returns an int,
            or None for the default number of points for NetworkHiddenActivations

    Returns:
        during_training (callable): see above
    """

    if num_points is None:
        num_points = lambda *args, **kwargs: None
    elif isinstance(num_points, int):
        _num_pts = num_points
        num_points = lambda *args, **kwargs: _num_pts
    elif not callable(num_points):
        raise ValueError(f'expected num_points is int or callable, got {num_points} (type={type(num_points)})')

    def during_training(savepath: str, train: bool,
                        digestor: typing.Optional[npmp.NPDigestor] = None,
                        measure_args: dict = None,
                        plot_args: dict = None):
        if not isinstance(savepath, str):
            raise ValueError(f'expected savepath is str, got {savepath} (type={type(savepath)})')
        if not isinstance(train, bool):
            raise ValueError(f'expected train is bool, got {train} (type={type(train)})')
        if digestor is not None and not isinstance(digestor, npmp.NPDigestor):
            raise ValueError(f'expected digestor is NPDigestor, got {digestor} (type={type(digestor)})')
        if measure_args is not None and not isinstance(measure_args, dict):
            raise ValueError(f'expected measure_args is dict, got {measure_args} (type={type(measure_args)})')
        if plot_args is not None and not isinstance(plot_args, dict):
            raise ValueError(f'expected plot_args is dict, got {plot_args} (type={type(plot_args)})')

        if measure_args is None:
            measure_args = dict()
        if plot_args is None:
            plot_args = dict()

        _, ext = os.path.splitext(savepath)
        if ext != '':
            raise ValueError(f'expected savepath has no ext, but got {savepath} (ext={ext})')
        if os.path.exists(savepath):
            raise FileExistsError(f'savepath {savepath} already exists')

        os.makedirs(savepath)

        def on_step(context: GenericTrainingContext, fname_hint: str):
            context.logger.info('[%s] - during training with hint %s', identifier, fname_hint)
            pwl = context.train_pwl if train else context.test_pwl
            outfile = os.path.join(savepath, f'{fname_hint}.zip')

            if isinstance(context.model, FeedforwardNetwork):
                hacts = get_hidacts_ff(context.model, pwl, num_points(context))
            elif isinstance(context.model, NaturalRNN):
                hacts = get_hidacts_rnn(context.model, pwl, num_points(context))
            else:
                raise ValueError(f'expected model is FeedforwardNetwork or NaturalRNN, got {context.model} (type={type(context.model)})')

            if 'xlabel' not in plot_args:
                plot_args['xlabel'] = hacts.label

            hacts.numpy()
            if digestor is not None:
                digestor(hacts.sample_points, hacts.sample_labels,
                         *hacts.hid_acts, outfile=outfile, netstyle=hacts.netstyle,
                         measure_kwargs=measure_args, plot_kwargs=plot_args,
                         target_module=target_module, target_name='digest')
            else:
                traj = measure(hacts, **measure_args)
                plot(traj, outfile, **plot_args)

        return on_step

    return during_training

def verify_ndarray(arr: np.ndarray, arr_name: str,
                   shape: typing.Optional[typing.Tuple[typing.Optional[int]]],
                   dtype: typing.Optional[str]):
    """Verifies that the given array is in fact an ndarray with the given
    shape and dtype.

    Args:
        arr (np.ndarray): the array that you want to verify
        arr_name (str): the name for the array that the caller sees
        shape (optional tuple of optional ints): the expected shape of the tuple.
            If you expect the shape to have length 2 but you do not know anything
            about the values, (None, None). If you expect the shape to have length
            2 and know the first is 5, (5, None), for length 1 but unknown (None,),
            etc.

            May use a tuple instead of a single int for each element, i.e.,
            (('num_layers', 5), None) for expecting
        dtype (typing.Optional[str]): The eexpected dtype of the array, either
            'float' or 'int'. Will be translated into the correct numpy classes
    """
    if not isinstance(arr_name, str):
        raise ValueError(f'expected arr_name is str, got {arr_name} (type={type(arr_name)})')
    if not isinstance(arr, np.ndarray):
        raise ValueError(f'expected {arr_name} is ndarray, got {arr} (type={type(arr)})')

    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            raise ValueError(f'expected shape is None, tuple, or list but got {shape} (type={type(shape)})')
        pretty_shape_builder = []
        for idx, size in enumerate(shape):
            if size is not None and not isinstance(size, (int, tuple, list)):
                raise ValueError(f'expected shape is list of None or int or tuple, but shape[{idx}] = {size} (type={type(size)})')

            if size is None:
                pretty_shape_builder.append('any')
            elif isinstance(size, int):
                pretty_shape_builder.append(str(size))
            else:
                if len(size) != 2:
                    raise ValueError(f'expected shape is a list of None, int, or tuple[exp_name, exp_size] but shape[{idx}] = {size} (bad length)')
                if not isinstance(size[0], str):
                    raise ValueError(f'expected shape is a list of None, int, or tuple[exp_name, exp_size] but shape[{idx}][0] = {size[0]} (type={type(size[0])} instead of str)')
                if size[1] is not None and not isinstance(size[1], int):
                    raise ValueError(f'expected shape is a list of None, int, or tuple [exp_name, exp_size] but shape[{idx}][1] = {size[1]} (type={type(size[1])} instead of int)')
                if size[1] is None:
                    pretty_shape_builder.append(size[0])
                else:
                    pretty_shape_builder.append(f'{size[0]}={size[1]}')

        pretty_shape = '[' + ', '.join(pretty_shape_builder) + ']'
        if len(arr.shape) != len(shape):
            raise ValueError(f'expected {arr_name}.shape is {pretty_shape}, got {arr.shape} (wrong num dims)')

        for idx, size in enumerate(shape):
            if size is not None:
                if isinstance(size, int):
                    if arr.shape[idx] != size:
                        raise ValueError(f'expected {arr_name}.shape is {pretty_shape}, got {arr.shape} (bad dim {idx})')
                elif size[1] is not None:
                    if arr.shape[idx] != size[1]:
                        raise ValueError(f'expected {arr_name}.shape is {pretty_shape}, got {arr.shape} (bad dim {idx})')

    if dtype is not None:
        if dtype == 'float':
            if arr.dtype not in (np.float, np.float32, np.float64):
                raise ValueError(f'expected {arr_name}.dtype is float-like, got {arr.dtype}')
        elif dtype == 'int':
            if arr.dtype == np.uint8:
                raise ValueError(f'uint8 style arrays are prone to issues, use int32 or int64')
            if arr.dtype not in (np.uint8, np.int, np.int32, np.int64):
                raise ValueError(f'expected {arr_name}.dtype is int-like, got {arr.dtype}')
        else:
            raise ValueError(f'expected the expected dtype is float or int, got {dtype}')



