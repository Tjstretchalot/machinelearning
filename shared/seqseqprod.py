"""Describes something that is capable of generating sequence to sequence problems to
present to a (recurrent) neural network"""

import shared.typeutils as tus
import typing
from shared.perf_stats import PerfStats, NoopPerfStats

class Sequence(typing.NamedTuple):
    """Describes an arbitrary sequence of collections of floats"""
    raw: typing.List[typing.Tuple[float]]


class SeqSeqProducer:
    """The abstract base class that sequence sequence producers inherit from

    Attributes:
        epoch_size (int): the number of points in an epoch
        input_dim (int): the number of inputs presented at each step (not the number of input steps!)
        output_dim (int): the number of outputs presented at each step (not the number of output steps!)

        __position (int): where we are in the dataset
        mark_stack (list): the stack of marks we can return to
    """

    def __init__(self, epoch_size: int, input_dim: int, output_dim: int):
        tus.check(epoch_size=(epoch_size, int), input_dim=(input_dim, int), output_dim=(output_dim, int))
        self.epoch_size = epoch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__position = 0
        self.mark_stack = []

    def __next__(self) -> typing.Tuple[Sequence, Sequence]:
        """Determines the next input/output sequence and moves forward"""
        result = self.get_current()
        if self.position >= self.epoch_size - 1:
            self.position = 0
        else:
            self.position = self.position + 1
        return result

    def get_current(self,
                    perf_stats: PerfStats = NoopPerfStats()) -> typing.Tuple[Sequence, Sequence]:
        """Gets the value at the current position. Returns the input and the output in that order"""
        raise NotImplementedError()

    @property
    def remaining_in_epoch(self) -> int:
        """Determines the number of values remaining in the current epoch"""
        return self.epoch_size - self.__position

    @property
    def position(self) -> int:
        """Gets/sets where this is inside the epoch. 0 corresponds to the beginning of the epoch"""
        return self.__position

    @position.setter
    def position(self, value: int) -> None:
        if value >= self.epoch_size:
            raise ValueError(f'position({value}) - not enough points (epoch_size = {self.epoch_size})')
        if value < 0:
            raise ValueError(f'position({value}) - cannot move to negative position')
        self._position(value)
        self.__position = value

    def mark(self):
        """Marks the current position such that it may be returned to via reset()"""
        self.mark_stack.append(self.__position)
        return self

    def reset(self):
        """Returns to the most recent mark"""
        if not self.mark_stack:
            return ValueError('no mark to return to!')
        self.position = self.mark_stack.pop()
        return self

    def _position(self, pos: int) -> None:
        """Moves to the specified position inside the epoch. Should not modify the position
        attribute. You may simply pass and use the self.position value

        Arguments:
            pos (int): the position inside the epoch to go to
        """
        raise NotImplementedError()