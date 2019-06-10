"""This module allows the bot to consider surprising events in the past
whenever it has some downtime.
"""
from sortedcontainers import SortedKeyList
from collections import deque
import typing
import time
import math
import sys
import random

class OfflineLearner:
    """An offline method for evaluating surprising results that the bot
    has encountered

    Attributes:
        surprising (SortedList[(loss, args, kwargs)]): surprising events we have seen recently,
            ordered from [0] being the most surprising to [-1] being the least surprising

        heap_size (int): max size we let the heap get to

        callback (callable[inputs] -> float): takes the inputs and returns

        samples (int): size of our rolling window

        random_skip_factor (float): the probability that we just skip over an element when training
            (0-1, 1 excluded)
        sum_roll_loss (float): sum(rolling_loss, 0)
        sum_roll_loss_sqd (float): sum((x*x for x in rolling_loss), 0)
        rolling_loss (deque[float]): the rolling loss

    """
    def __init__(self, callback: typing.Callable, heap_size: int = 10, samples: int = 100,
                 random_skip_factor: float = 0.7):
        self.surprising = SortedKeyList(key=lambda x: -x[0])
        self.callback = callback
        self.heap_size = heap_size

        self.samples = samples
        self.sum_roll_loss = 0
        self.sum_roll_loss_sqd = 0
        self.rolling_loss = deque()

        self.random_skip_factor = random_skip_factor

    def __call__(self, *args, **kwargs) -> None:
        """Registers the given arguments and keyword arguments with the given
        loss"""
        loss = self.callback(*args, **kwargs)
        self._handle(loss, 1, args, kwargs)

    def _handle(self, loss, counter, args, kwargs):
        if counter > 5:
            return
        if counter == 1:
            popped = self.rolling_loss.popleft() if len(self.rolling_loss) >= self.samples else 0
            self.rolling_loss.append(loss)

            self.sum_roll_loss += loss - popped
            self.sum_roll_loss_sqd += loss*loss - popped*popped

        if len(self.rolling_loss) < self.samples:
            return

        meas_first = self.sum_roll_loss / len(self.rolling_loss)
        meas_second = self.sum_roll_loss_sqd / len(self.rolling_loss)
        variance = meas_second - (meas_first**2)
        std_dev = math.sqrt(variance)

        if -2 * std_dev < loss - meas_first < 2 * std_dev:
            return

        if counter == 1:
            print(f'[offline] found something surprising: {loss} '
                  + f'(mean: {self.sum_roll_loss / self.samples}, '
                  + f'vari: {variance}, std: {std_dev})')
            sys.stdout.flush()


        if len(self.surprising) >= self.heap_size:
            if loss < self.surprising[-1][0]:
                return
            self.surprising.pop()
        self.surprising.add((loss, args, kwargs, counter))

    def think(self, maxtime: float):
        """Thinks for the specified amount of time by invoking the callback with
        the arguments passed to register in order of most to least surprising
        """

        end = time.time() + maxtime
        max_todo = len(self.surprising)
        done = 0
        to_add = []
        while time.time() < end and self.surprising:
            loss, args, kwargs, counter = self.surprising.pop(0)
            if random.random() < self.random_skip_factor:
                to_add.append((loss, args, kwargs, counter))
                done += 1
                continue

            loss = self.callback(*args, **kwargs)
            self._handle(loss, counter + 1, args, kwargs)
            done += 1
            if done >= max_todo:
                break

        for val in to_add:
            self.surprising.add(val)
