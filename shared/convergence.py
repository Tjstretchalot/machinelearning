"""This module provides an interface and various implementations for an
object which is capable of determining if the liminf, limsup, or limit
likely exists *and* has been reached to within some epsilon for a given
sequence.

For example, the sequence -5, -4, -3, -2, -1, ... would fail to ever
have a convergent liminf, limsup, or limit.

A bounded sinusoidal sequence could have a liminf and limsup but no limit.

If a sinusoidal sequence has peaks which are unbounded and troughs which are
bounded, it might have a liminf and no limsup
"""

import typing

class ConvergenceChecker:
    """Describes something which can determine if a sequence is converging.

    Attributes:
        epsilon (float): how close we are trying to estimate the given limit
    """

    @property
    def style(self) -> str:
        """One of the following: 'liminf', 'limsup', 'limit'. Describes what
        this checker is looking for, and will not typically have a setter.
        """
        raise NotImplementedError

    @property
    def liminf(self) -> bool:
        """Returns true if the style of this checker is 'liminf', returns
        false otherwise"""
        return self.style == 'liminf'

    @property
    def limsup(self) -> bool:
        """Returns true if the style of this checker is 'limsup', returns
        false otherwise"""
        return self.style == 'limsup'

    @property
    def limit(self) -> bool:
        """Returns true if the style of this checker is 'limit', returns
        false otherwise"""
        return self.style == 'limit'

    def __init__(self, epsilon: float):
        if not isinstance(epsilon, float):
            raise ValueError(f'expected epsilon is float, got {epsilon} (type={type(epsilon)})')
        if epsilon < 0:
            raise ValueError(f'expected epsilon is nonnegative, got {epsilon:.2e}')

        self.epsilon = epsilon

    def stats(self) -> str:
        """Returns a human-readable description of the internal state of this
        convergence checker. For example, this could include the memory mean
        and standard deviation.
        """
        raise NotImplementedError

    def push(self, value: float) -> None:
        """Pushes a single value to this convergence checker.
        """
        raise NotImplementedError

    def converged(self) -> typing.Tuple[bool, typing.Optional[float]]:
        """Determines if the given limit (self.style) has converged to within
        epsilon of the target. If it has, returns True and the value it has
        converged to. If it has not, returns False, None
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clears any internal storage that this has. Equivalent to reconstructing."""
        raise NotImplementedError

class PatienceConvergenceChecker(ConvergenceChecker):
    """A convergence checker that will wait a particular amount of time to
    see an change of epsilon for either the supremum or infimum (depending
    on how it was initialized). This requires a strong assumption about the
    underlying sequence: If we are looking for the limsup, then the sequence
    is bounded above by the limsup. Alternatively, if we are looking for the
    liminf, then the sequence is bounded below by the liminf.

    This convergence checker requires a tiny amount of finite memory. It is
    a strong candidate for determining if loss is non-decreasing.

    Attributes:
        patience (int): how long we will wait without seeing any improvement.
        best (float): the best value we have seen for the limit we are looking
            for.
        patience_used (int): how much time we have weighted without seeing any
            improvement
    """

    def __init__(self, patience: int, style='liminf', epsilon=1e-6):
        super().__init__(epsilon)

        if not isinstance(patience, int):
            raise ValueError(f'expected patience is int, got {patience} (type={type(patience)})')
        if style not in ('liminf', 'limsup'):
            raise ValueError(f'expected style is liminf or limsup, got {style}')

        self._style = style
        self.patience = patience
        self.best = float('inf') if self.liminf else float('-inf')
        self.patience_used = 0

    @property
    def style(self) -> str:
        return self._style

    def stats(self) -> str:
        return f'best={self.best:.3e} (patience: {self.patience_used}/{self.patience})'

    def push(self, value: float) -> None:
        better = False
        if self.liminf:
            better = value + self.epsilon < self.best
        else:
            better = value - self.epsilon > self.best

        if better:
            self.best = value
            self.patience_used = 0
        else:
            self.patience_used += 1

    def converged(self) -> bool:
        return self.patience_used >= self.patience

    def reset(self) -> None:
        self.best = float('inf') if self.liminf else float('-inf')
        self.patience_used = 0