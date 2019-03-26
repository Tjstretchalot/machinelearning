"""Describes something which can produce points with labels"""

import typing
import torch

class PointWithLabel(typing.NamedTuple):
    """Describes a simple point with a label

    Attributes:
        point (torch.tensor): the point
        label (int): the label
    """

    point: torch.tensor
    label: int

class PointWithLabelProducer:
    """Describes something which can produce points with labels. Supports
    iteration by giving PointWithLabel's

    Attributes:
        epoch_size (int): the size of an epoch
        input_dim (int): the dimension of the points returned
        output_dim (int): the labels are in range(output_dim)

        __position (int): where we are within the epoch
        mark_stack (list): the stack of marks we can return to
    """

    def __init__(self, epoch_size: int, input_dim: int, output_dim: int):
        self.epoch_size = epoch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__position = 0
        self.mark_stack = []

    @property
    def remaining_in_epoch(self) -> int:
        """Returns the number of points remaining in the epoch"""
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

    def fill(self, points: torch.tensor, labels: torch.tensor) -> None:
        """Fills the given points array, which should have shape [batch_size x input_dim]
        and associates each point with the label by having the same index. For example,
        points[0] has labels[0].

        Arguments:
            points (torch.tensor, [batch_size x input_dim] dtype=torch.double or torch.float)
            labels (torch.tensor, [batch_size] dtype=torch.long or torch.uint8)
        """
        if len(points.shape) != 2:
            raise ValueError(f'expected points has shape [batch_size x input_dim], got {points.shape}')
        if len(labels.shape) != 1:
            raise ValueError(f'expected labels has shape [batch_size], got {labels.shape}')
        if points.shape[0] != labels.shape[0]:
            raise ValueError('expected points [batch_size x input_dim] and labels [batch_size] to'
                             + f' have same first dimension, but points.shape[0]={points.shape[0]}'
                             + f' and labels.shape[0]={labels.shape[0]}')

        batch_size = points.shape[0]
        avail = self.remaining_in_epoch
        if avail > batch_size:
            self._fill(points, labels)
            self.position += batch_size
        elif avail == batch_size:
            self._fill(points, labels)
            self.position = 0
        else:
            self._fill(points[:avail], labels[:avail])
            self.position = 0
            self.fill(points[avail:], labels[avail:])

    def __next__(self) -> PointWithLabel:
        points = torch.zeros((1, self.input_dim), dtype=torch.double)
        labels = torch.zeros(1, dtype=torch.long)
        self.fill(points, labels)
        return PointWithLabel(point=points[0], label=labels[0])

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

    def _fill(self, points: torch.tensor, labels: torch.tensor) -> None:
        """Fills the given points array, which should have shape [batch_size x input_dim]
        and associates each point with the label by having the same index. For example,
        points[0] has labels[0]. This is guarranteed to not overflow the epoch, i.e.,
        this will only be called if (epoch_size - position - points.shape[0]) is nonnegative

        Arguments:
            points (torch.tensor, [batch_size x input_dim] dtype=torch.double or torch.float)
            labels (torch.tensor, [batch_size] dtype=torch.long or torch.uint8)
        """
        raise NotImplementedError()

    def _position(self, pos: int) -> None:
        """Moves to the specified position inside the epoch. Should not modify the position
        attribute. You may simply pass and use the self.position value

        Arguments:
            pos (int): the position inside the epoch to go to
        """
        raise NotImplementedError()

class SimplePointWithLabelProducer(PointWithLabelProducer):
    """Describes a point with label producer that uses a given backing tensor for data

    Attributes:
        real_points (torch.tensor [epoch_size x input_dim])
        real_labels (torch.tensor [epoch_size])
    """

    def __init__(self, real_points: torch.tensor, real_labels: torch.tensor, output_dim: int):
        super().__init__(real_points.shape[0], real_points.shape[1], output_dim)

        self.real_points = real_points
        self.real_labels = real_labels

    def _fill(self, points: torch.tensor, labels: torch.tensor) -> None:
        pos = self.position
        points[:] = self.real_points[pos:pos + points.shape[0]]
        labels[:] = self.real_labels[pos:pos + labels.shape[0]]

    def _position(self, pos: int) -> None:
        pass
