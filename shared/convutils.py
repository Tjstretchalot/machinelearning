"""This module makes it easier to work with convolutional layers"""

import torch
import torch.nn
from functools import reduce
import operator
from shared.models.ff import ComplexLayer

class Reshape(torch.nn.Module):
    """Simple reshaping layer"""
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, *args):
        """Changes the view of x to the desired shape"""
        return args[0].view(self.shape)

def flatten_after_maxpool(hidacts):
    """A valid operation on the hidacts that flattens after maxpooling"""
    return hidacts.reshape(-1, reduce(operator.mul, hidacts.shape[1:]))

def unflatten_to(*args):
    """Calls reshape on the hidacts with the given args"""
    return Reshape(*args)

class FluentShape:
    """a fluent api for reshaping from maxpooling

    Attributes:
        dims (n-tuple): the shape we currently have.
        _verbose (bool): true to print out net structure as its created, false not to
            (only for inplace operations)
    """

    def __init__(self, dims, verbose=False):
        if isinstance(dims, int):
            self.dims = (dims,)
        elif isinstance(dims, (list, torch.Size)):
            self.dims = tuple(dims)
        elif isinstance(dims, tuple):
            self.dims = dims
        else:
            raise ValueError(f'expected dims is tuple, got {dims} (type={type(dims)})')

        for idx, val in enumerate(self.dims):
            if not isinstance(val, int):
                raise ValueError(f'expected dims[{idx}] is int, got {val} (type={type(val)})')

        self._verbose = verbose

    def verbose(self):
        """Returns a copy of this fluent shape that has verbose set to True"""
        return FluentShape(self.dims, True)

    def quiet(self):
        """Returns a copy of this fluent shape that has verbose set to False"""
        return FluentShape(self.dims, False)

    def copy_(self, other):
        """Copies the dims from the given fluent shape into this one and returns this"""
        if isinstance(other, FluentShape):
            self.dims = other.dims
            self._verbose = other.verbose
        elif isinstance(other, tuple):
            self.dims = other
        elif isinstance(other, (list, torch.Size)):
            self.dims = tuple(other)
        elif isinstance(other, int):
            self.dims = (other,)
        else:
            raise ValueError(f'unknown fluentshape style: {other} (type={type(other)})')

        for idx, val in enumerate(self.dims):
            if not isinstance(val, int):
                raise ValueError(f'expected other[{idx}] is int, got {val} (type={type(val)})')

        return self

    def unflatten_conv(self, channels, width, height):
        """Unflattens a flattened shape into a convolvable one with the given number
        of channels, width, and height"""
        if len(self.dims) != 1:
            raise ValueError(f'cannot unflatten non-flat {self.dims}')

        if self.dims[0] != channels*width*height:
            raise ValueError(f'cannot unflatten {self.dims} to {channels}, {width}, {height} (flattened that is {channels*width*height})')

        return FluentShape((channels, height, width), self.verbose)

    def unflatten_conv_(self, channels, width, height):
        """In-place unflatten for convolution and return the appropriate complex layer
        that performs this operation
        """
        newme = self.unflatten_conv(channels, width, height)
        lyr = ComplexLayer(
            style='other', is_module=False, invokes_callback=False,
            action=unflatten_to(-1, channels, width, height))

        if self._verbose:
            print(f'unflatten_conv_({channels}, {width}, {height}) -> {newme.dims}')

        self.copy_(newme)
        return lyr

    def unflatten_conv3(self, channels, depth, width, height):
        """Unflattens a flattened shape into a 3d convolvable one with the given
        number of channels, depth, width, and height"""
        if len(self.dims) != 1:
            raise ValueError(f'cannot unflatten non-flat {self.dims}')

        if self.dims[0] != channels*depth*width*height:
            raise ValueError(f'cannot unflatten {self.dims} to {channels}, {depth}, {width}, {height} (flattened that is {channels*depth*width*height})')

        return FluentShape((channels, depth, width, height), self.verbose)

    def unflatten_conv3_(self, channels, depth, width, height):
        """In-place unflatten to given convolvable setting and returns a ComplexLayer which
        performs the necessary operation to unflatten"""
        newme = self.unflatten_conv3(channels, depth, width, height)
        lyr = ComplexLayer(
            style='other', is_module=False, invokes_callback=False,
            action=unflatten_to(-1, channels, depth, width, height))
        self.copy_(newme)

        if self._verbose:
            print(f'unflatten_conv3_({channels}, {depth}, {width}, {height}) -> {newme.dims}')

        return lyr

    def conv(self, out_channels, kernel_size, stride=1, padding=0):
        """Returns a new fluent shape that is the convolution of this with the given
        kernel size"""
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        if len(self.dims) != 3:
            raise ValueError(f'cannot convolve this without reshaping to (C_in, H, W)')

        kernel_width, kernel_height = kernel_size
        stride_width, stride_height = stride
        padding_width, padding_height = padding
        _, height, width = self.dims

        if kernel_height > height:
            raise ValueError(f'kernel size cannot exceed height of {height} (got {kernel_height})')
        if kernel_width > width:
            raise ValueError(f'kernel size cannot exceed width of {width} (got {kernel_width})')

        out_height = int((height - kernel_height + 2 * padding_height)/stride_height) + 1
        out_width = int((width - kernel_width + 2 * padding_width)/stride_width) + 1

        return FluentShape((out_channels, out_height, out_width), self.verbose)

    def conv_(self, out_channels, kernel_size, stride=1, padding=0, invokes_callback=False):
        """In-place convolve dimensions and return appropriate ComplexLayer"""

        newme = self.conv(out_channels, kernel_size, stride, padding)
        lyr = ComplexLayer(
            style='layer', is_module=True, invokes_callback=invokes_callback,
            action=torch.nn.Conv2d(
                in_channels=self.dims[0],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))

        if self._verbose:
            print(f'conv_({out_channels}, {kernel_size}, stride={stride}, padding={padding}, invokes_callback={invokes_callback}) -> {newme.dims}')

        self.copy_(newme)
        return lyr

    def conv3(self, out_channels, kernel_size, stride=1, padding=0):
        """Returns a new fluent shape that is the 3D convolution of this with the given
        kernel size"""
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        if len(self.dims) != 4:
            raise ValueError(f'cannot 3d convolve {self.dims} (should be Cin, D, W, H)')

        kdep, khei, kwid = kernel_size
        sdep, shei, swid = stride
        pdep, phei, pwid = padding

        _, depth, width, height = self.dims

        if kdep > depth:
            raise ValueError(f'kernel size cannot exceed depth of {depth} (got {kdep})')
        if khei > height:
            raise ValueError(f'kernel size cannot exceed height of {height} (got {khei})')
        if kwid > width:
            raise ValueError(f'kernel size cannot exceed width of {width} (got {kwid})')

        out_depth = int((depth - kdep + 2 * pdep)/sdep) + 1
        out_height = int((height - khei + 2 * phei)/shei) + 1
        out_width = int((width - kwid + 2 * pwid)/swid) + 1

        return FluentShape((out_channels, out_depth, out_height, out_width), self.verbose)

    def conv3_(self, out_channels, kernel_size, stride=1, padding=0, invokes_callback=False):
        """In-place 3d convolution and return the complex layer to do it"""
        newme = self.conv3(out_channels, kernel_size, stride, padding)
        lyr = ComplexLayer(
            style='layer', is_module=True, invokes_callback=invokes_callback,
            action=torch.nn.Conv3d(
                in_channels=self.dims[0],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))

        if self._verbose:
            print(f'conv3_({out_channels}, {kernel_size}, stride={stride}, padding={padding}, invokes_callback={invokes_callback}) -> {newme.dims}')

        self.copy_(newme)
        return lyr

    def maxpool(self, kernel_size, stride=1):
        """Returns a new fluent shape that is this maxpooled with the given kernel size and
        stride"""
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        if len(self.dims) != 3:
            raise ValueError(f'need to have dim (C, H, W) to maxpool but have dim {self.dims}')

        channels, height, width = self.dims
        kernel_height, kernel_width = kernel_size
        stride_height, stride_width = stride

        if kernel_height > height:
            raise ValueError(f'kernel size cannot exceed height of {height} (got {kernel_height})')
        if kernel_width > width:
            raise ValueError(f'kernel size cannot exceed width of {width} (got {kernel_width})')

        out_height = int((height - kernel_height) / stride_height) + 1
        out_width = int((width - kernel_width) / stride_width) + 1

        return FluentShape((channels, out_height, out_width), self.verbose)

    def maxpool_(self, kernel_size, stride=1):
        """Inplace maxpool2d and return appropriate ComplexLayer"""
        newme = self.maxpool(kernel_size, stride)
        lyr = ComplexLayer(
            style='layer', is_module=True, invokes_callback=False,
            action=torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

        if self._verbose:
            print(f'maxpool_({kernel_size}, stride={stride}) -> {newme.dims}')

        self.copy_(newme)
        return lyr

    def maxpool3(self, kernel_size, stride=1):
        """Returns a new fluent shape that is the 3d maxpooling of this with the given
        kernel size and stride"""
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        if len(self.dims) != 4:
            raise ValueError(f'need to have dim (C, D, H, W) to maxpool3, have dim {self.dims}')

        channels, depth, height, width = self.dims
        kdep, khei, kwid = kernel_size
        sdep, shei, swid = stride

        if kdep > depth:
            raise ValueError(f'kernel size cannot exceed depth of {depth} (got {kdep})')
        if khei > height:
            raise ValueError(f'kernel size cannot exceed height of {height} (got {khei})')
        if kwid > width:
            raise ValueError(f'kernel size cannot exceed width of {width} (got {kwid})')

        out_depth = int((depth - kdep) / sdep) + 1
        out_height = int((height - khei) / shei) + 1
        out_width = int((width - kwid) / swid) + 1

        return FluentShape((channels, out_depth, out_height, out_width), self.verbose)

    def maxpool3_(self, kernel_size, stride=1):
        """In-place maxpool3 and return the complex layer to do it"""
        newme = self.maxpool3(kernel_size, stride)
        lyr = ComplexLayer(
            style='layer', is_module=True, invokes_callback=False,
            action=torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride))

        if self._verbose:
            print(f'maxpool3_({kernel_size}, stride={stride}) -> {newme.dims}')

        self.copy_(newme)
        return lyr

    def flatten(self):
        """Returns a flattened version of this shape"""
        return FluentShape(reduce(operator.mul, self.dims), self.verbose)

    def flatten_(self, invokes_callback=False):
        """Flattens this shape in place and returns the appropriate layer to flatten
        with"""
        newme = self.flatten()
        lyr = ComplexLayer(
            style='other', is_module=False, invokes_callback=invokes_callback,
            action=flatten_after_maxpool
        )
        if self._verbose:
            print(f'flatten_(invokes_callback={invokes_callback}) -> {newme.dims}')

        self.copy_(newme)
        return lyr

    def linear_(self, output_dim, invokes_callback=False):
        """Inplace linear layer to output dim and return appropriate ComplexLayer"""
        if not isinstance(output_dim, int):
            raise ValueError(f'expected output_dim is int, got {output_dim} (type={type(output_dim)})')
        if len(self.dims) != 1:
            raise ValueError(f'cannot do all-to-all when not flat (have dims {self.dims})')

        lyr = ComplexLayer(style='layer', is_module=True, invokes_callback=invokes_callback,
               action=torch.nn.Linear(self.dims[0], output_dim))
        self.dims = (output_dim,)

        if self._verbose:
            print(f'linear_({output_dim}, invokes_callback={invokes_callback}) -> {self.dims}')

        return lyr

    def tanh(self, invokes_callback=True):
        """Convenience function that produces a tanh nonlinearity complex layer"""
        if self._verbose:
            print(f'tanh(invokes_callback={invokes_callback}) -> {self.dims}')

        return ComplexLayer(
            style='nonlinearity', is_module=False,
            invokes_callback=invokes_callback,
            action=torch.tanh)

    def relu(self, invokes_callback=True):
        """Convenience function that produces a relu nonlinearity complex layer"""
        if self._verbose:
            print(f'relu(invokes_callback={invokes_callback}) -> {self.dims}')

        return ComplexLayer(
            style='nonlinearity', is_module=False,
            invokes_callback=invokes_callback,
            action=torch.relu)
