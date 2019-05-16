"""This module provides the interface and implementations for weight-initialization
functions, including with reasonably fast serialization.
"""

import torch
import math

class WeightInitializer:
    """Interface for something which is capable of initializing the weights of a network
    """

    @classmethod
    def identifier(cls):
        """Returns a string which uniquely corresponds to this weight initializer and is
        in snake case.
        """

        raise NotImplementedError()

    def initialize(self, weights):
        """Initializes the given weights according to how this initializer was constructed.
        The weights are assumed to start off as all zeros. The dimensions of the tensor
        depend on the initializer being used.

        Args:
            weights (tensor): The tensor to initialize according to the method
        """

        raise NotImplementedError()

    def serialize(self):
        """Serializes this weight initializer such that it can be deserialized using
        the module function 'deserialize'
        """

        return (self.identifier(), self._serialize())

    def _serialize(self):
        """Returns a primitive or tuple. A tuple may contain other tuples or primitives.
        Does not include the identifier
        """

        raise NotImplementedError()

    @classmethod
    def deserialize(cls, serd):
        """Deserializes the instance of this class serialized with _serialize. Note that
        this does NOT correspond to deserializing the result of serialize, which must be
        used to determine the class type before calling this.

        Args:
            serd (any): the result from _serialize
        """

        raise NotImplementedError()

WEIGHT_INIT_IMPLS = dict()

class ZerosWeightInitializer(WeightInitializer):
    """No-op. Useful for biases"""

    def __init__(self):
        pass

    @classmethod
    def identifier(cls):
        return 'zeros'

    def initialize(self, weights):
        pass

    def _serialize(self):
        return tuple()

    @classmethod
    def deserialize(cls, serd):
        return cls()

WEIGHT_INIT_IMPLS[ZerosWeightInitializer.identifier()] = (
    ZerosWeightInitializer
)

class OrthogonalWeightInitializer(WeightInitializer):
    """
    Initialize the weight matrix to preserve the dimensionality and 2-norm according to the gain

    The underlying geometry of the matrix is preserved

    Attributes:
        gain (float): Scaling factor for the norm of the matrix
        normalize_dim (int, optional): if set, the gain is divided by the sqrt of the given
                                       index in the shape of the weight matrix.
    """

    def __init__(self, gain, normalize_dim=None):
        self.gain = gain
        self.normalize_dim = normalize_dim

    @classmethod
    def identifier(cls):
        return 'orthogonal'

    def initialize(self, weights):
        gain = self.gain
        if self.normalize_dim is not None:
            gain /= math.sqrt(weights.shape[self.normalize_dim])

        torch.nn.init.orthogonal_(weights, gain)

    def _serialize(self):
        return (self.gain, self.normalize_dim)

    @classmethod
    def deserialize(cls, serd):
        return cls(serd[0], serd[1])

WEIGHT_INIT_IMPLS[OrthogonalWeightInitializer.identifier()] = (
    OrthogonalWeightInitializer
)

class GaussianWeightInitializer(WeightInitializer):
    """Initializes the entire weight matrix using only untruncated gaussians
    with a given mean and variance.

    Attributes:
        mean (float): the mean for the gaussian
        vari (float): the variance for the gaussian
        normalize_dim (int, optional): If set, the variance is divided by the square root of the
            shape along the given dimension
    """

    def __init__(self, mean=0, vari=1, normalize_dim=None):
        self.mean = mean
        self.vari = vari
        self.normalize_dim = normalize_dim

    @classmethod
    def identifier(cls):
        return 'gaussian'

    def initialize(self, weights):
        if self.vari == 0:
            weights += self.mean
            return

        torch.randn(weights.shape, out=weights)

        vari = self.vari
        if self.normalize_dim is not None:
            vari /= math.sqrt(weights.shape[self.normalize_dim])

        weights *= vari
        weights += self.mean

    def _serialize(self):
        return (self.mean, self.vari, self.normalize_dim)

    @classmethod
    def deserialize(cls, serd):
        return cls(*serd)

WEIGHT_INIT_IMPLS[GaussianWeightInitializer.identifier()] = GaussianWeightInitializer

class FixedEigenvalueRadiusWeightInitializer(WeightInitializer):
    """Initializes a square weight matrix to approximately result in a fixed radius of
    the disk of eigenvalues. This is equivalent to a a gaussian weight matrix with mean
    0 and variance g/sqrt(n)

    Attributes:
        radius (float): the target (real) radius for the disk of eigenvalues
    """

    def __init__(self, radius):
        self.radius = radius

    @classmethod
    def identifier(cls):
        return 'fixed_g_radius'

    def initialize(self, weights):
        size = weights.shape[0]
        if size != weights.shape[1]:
            raise ValueError(f'matrix not square; got [{size}, {weights.shape[1]}]')
        torch.randn(weights.shape, out=weights)
        weights *= self.radius / math.sqrt(size)
        return weights

    def _serialize(self):
        return self.radius

    @classmethod
    def deserialize(cls, serd):
        return cls(serd)

WEIGHT_INIT_IMPLS[FixedEigenvalueRadiusWeightInitializer.identifier()] = (
    FixedEigenvalueRadiusWeightInitializer)

class SompolinskyFixedGainWeightInitializer(WeightInitializer):
    """Initializes the weights of a square matrix to approximately result in a fixed amount
    of gain as if it were in a Sompolinsky style network.

    See Eqn. 5 in Kapmon and Sompolinsky - 2015 - Transition to chaos... (note that this is
    the variance, so we need to multiply by sqrt of that!)

    Attributes:
        gain (float): Target fixed amount of gain
    """

    def __init__(self, gain):
        self.gain = gain

    @classmethod
    def identifier(cls):
        return 'sompolinsky_fixed_gain'

    def initialize(self, weights):
        size = weights.shape[0]
        if weights.shape[1] != size:
            raise ValueError(f'matrix is not square: [{size}x{weights.shape[1]}]')
        torch.eye(size, out=weights)
        weights += torch.randn(size, size) * (self.gain / math.sqrt(size))

    def _serialize(self):
        return self.gain

    @classmethod
    def deserialize(cls, serd):
        return cls(serd)

WEIGHT_INIT_IMPLS[SompolinskyFixedGainWeightInitializer.identifier()] = (
    SompolinskyFixedGainWeightInitializer
)

class SompolinskySmoothedFixedGainWeightInitializer(WeightInitializer):
    """Initializes the weights of a square matrix using a linear approximation
    for the nonlinearity to be smooth, as if it were in a Sompolinsky style
    network with fixed gain.

    Attributes:
        smoothing_factor (float): A nonnegative float less than 1 that corresponds to the
            smoothness imposed on the weights
        gain (float): The desired gain on the network
    """

    def __init__(self, smoothing_factor, gain):
        self.smoothing_factor = smoothing_factor
        self.gain = gain

    @classmethod
    def identifier(cls):
        return 'sompolinsky_random_smoothed_fixed_gain'

    def initialize(self, weights):
        #(1 - dt) * torch.eye(N, N) + dt * g_radius * torch.randn(N, N) / np.sqrt(N)
        size = weights.shape[0]
        if weights.shape[1] != size:
            raise ValueError(f'matrix is not square: [{size}x{weights.shape[1]}]')
        torch.eye(size, out=weights)
        weights *= (1 - self.smoothing_factor)
        weights += torch.randn_like(weights) * (self.gain / math.sqrt(size)) * self.smoothing_factor

    def _serialize(self):
        return (self.smoothing_factor, self.gain)

    @classmethod
    def deserialize(cls, serd):
        return cls(serd[0], serd[1])

WEIGHT_INIT_IMPLS[SompolinskySmoothedFixedGainWeightInitializer.identifier()] = (
    SompolinskySmoothedFixedGainWeightInitializer
)

class RectangularEyeWeightInitializer(WeightInitializer):
    """Constructs the weight matrix by setting 0,0 to gain, 1,1 to gain, 2,2 to gain
    etc much like the eye function, except stopping at the shorter of the two dimensions
    if the weight matrix is not square.

    Attributes:
        gain (float): The value that the diagonal components are set to
        normalize_dim (int, optional): If set, the gain is divided by the square root
            of the weights shape along this dimension

    """

    def __init__(self, gain=1, normalize_dim=None):
        self.gain = float(gain)
        self.normalize_dim = normalize_dim

    @classmethod
    def identifier(cls):
        return "rectangular_eye"

    def initialize(self, weights):
        torch.eye(weights.shape[0], weights.shape[1], out=weights)

        gain = self.gain
        if self.normalize_dim is not None:
            gain /= math.sqrt(weights.shape[self.normalize_dim])

        weights *= gain

    def _serialize(self):
        return (self.gain, self.normalize_dim)

    @classmethod
    def deserialize(cls, serd):
        return cls(*serd)

WEIGHT_INIT_IMPLS[RectangularEyeWeightInitializer.identifier()] = (
    RectangularEyeWeightInitializer
)

def deserialize(serd_weight_init) -> WeightInitializer:
    """Deserializes the object serialized via serialize(), assuming that the object
    is a weight initializer.

    Args:
        serd_weight_init (typing.Any): The object that was returned from serialize

    Returns:
        WeightInitializer: An equivalent instance of a weight initializer
    """

    return WEIGHT_INIT_IMPLS[serd_weight_init[0]].deserialize(serd_weight_init[1])

def deser_or_noop(serd_weight_init) -> WeightInitializer:
    """If the object is a weight initializer already, returns the object. Otherwise,
    attempts to deserialize it.

    Args:
        serd_weight_init (typing.Any): either a weight initializer or a serialized one

    Returns:
        WeightInitializer: the weight initializer it is referring to
    """
    if isinstance(serd_weight_init, WeightInitializer):
        return serd_weight_init
    return deserialize(serd_weight_init)