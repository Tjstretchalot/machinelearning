"""This module handles performance statistics for modules. It is imperative that this
can be logged to file or easily printed and it minimizes the amount of useless information
displayed
"""
import typing
import numpy as np
from collections import deque
import time
import sys
from datetime import datetime

MAX_DEPTH = 100

class PerfStats:
    """Describes a performance tracking class. A single instance of a class tracks a single
    region but is nestable.

    Examples:

    ```py
        pstats = PerfStats() # not logging data
        pstats.enter('REGION_1')
        pstats.enter('SUBREGION_A')
        time.sleep(1)
        pstats.exit()
        for i in range(3):
            pstats.enter('SUBREGION_B')
            time.sleep(0.5)
            pstats.exit()
        pstats.exit()
        pstats.print()
    ```

    Attributes:
        identifier (str, optional): the identifier for this performance stats
        depth (int): how deep this layer is, where 0 is the top-level instance

        rolling_large (deque): each index is 2 items long
            used for outputting performance information. left-most entries are the oldest
            and right-most entries are the newest. every item is (sum_time, num_entries)
        rolling_time (float): the total time in rolling_large
        rolling_count (int): the total number of enters() in rolling_large

        cur_sum_start (float): the time.time() when we began the current batch. we batch items
            into a minimum intervals to avoid excessive deque usage at the cost of some granularity
        cur_sum_count (int): the number of items in the current sum
        cur_sum_time (float): the time (in seconds) that elapsed during the current sum
        cur_start (float, optional): the start time when enter was last called
            or None if not entered

        children (dict): a dict of children where the key is the region name and the value is
            PerfStats.
        active_child (PerfStats): the active child if there is one
    """

    def __init__(self, identifier: typing.Optional[str], depth: int = 0):
        self.identifier = identifier
        self.depth = depth

        self.rolling_large = deque()
        self.rolling_time = 0.0
        self.rolling_count = 0

        self.cur_sum_start = 0.0
        self.cur_sum_count = 0
        self.cur_sum_time = 0.0
        self.cur_start = None

        self.children = dict()
        self.active_child = None

    def _start(self):
        if self.cur_start is not None:
            raise RuntimeError(f'cannot start {self.identifier} (already started)')

        the_time = time.time()
        if self.cur_sum_count == 0:
            self.cur_sum_start = the_time
        self.cur_start = the_time

    def _end(self):
        if self.cur_start is None:
            raise RuntimeError(f'cannot end {self.identifier} (not started)')

        the_time = time.time()
        duration = the_time - self.cur_start
        self.cur_start = None
        self.cur_sum_time += duration
        self.cur_sum_count += 1

        batch_time = the_time - self.cur_sum_start
        if batch_time > 1:
            self._batch()

    def _batch(self):
        if len(self.rolling_large) > 100:
            pop_time, pop_num = self.rolling_large.popleft()
            self.rolling_count -= pop_num
            self.rolling_time -= pop_time

        self.rolling_large.append((self.cur_sum_time, self.cur_sum_count))
        self.rolling_count += self.cur_sum_count
        self.rolling_time += self.cur_sum_time
        self.cur_sum_time = 0
        self.cur_sum_count = 0

    def enter(self, region_name):
        """Enters the region denoted by the specified name

        Args:
            region_name (str): the name of the region to enter, typically caps and underscores
        """
        if self.active_child is not None:
            self.active_child.enter(region_name)
            return

        if region_name not in self.children:
            if self.depth > MAX_DEPTH:
                raise RuntimeError(f'exceeded max depth with region {region_name}')
            self.children[region_name] = PerfStats(region_name, self.depth + 1)

        self.children[region_name]._start() # pylint: disable=protected-access

    def exit(self) -> typing.Tuple[bool, bool]:
        """Closes the most recent region entered. The first result is True if we closed
        this region and false otherwise. Will always be false for top region. The second
        result is True if we closed our child and False otherwise.
        """
        if self.active_child:
            if self.active_child.exit()[0]:
                self.active_child = None
                return False, True
            return False, False

        self._end()
        return True, False

    def mean(self):
        """Determines the mean time spent in this region. Returns 0 if this region is not
        entered (i.e. the top-level region).
        """
        if self.rolling_count == 0:
            return 0
        return self.rolling_time / self.rolling_count

    def print(self, out=None, level='info', indent=''):
        """Prints the performance hints to the given filehandle or sys.stdout if
        one is not provided. If out is a str, it is interpreted as a file

        Args:
            out (filehandle, optional): Defaults to None. Where to print performance information
            level (str): one of 'trace', 'debug', 'info' - determines the level of information
                printed
            indent (str): prepended to every log line
        """
        if isinstance(out, str):
            with open(out, 'w+') as outfile:
                self.print(outfile)
            return

        if out is None:
            out = sys.stdout

        if level == 'trace':
            num_hotspots = float('inf')
        elif level == 'debug':
            num_hotspots = 5
        elif level == 'info':
            num_hotspots = 2

        child_names = np.zeros(len(self.children), dtype='object')
        child_means = np.zeros(len(self.children), dtype='float64')

        for (i, (k, child)) in enumerate(self.children.items()):
            child_names[i] = k
            child_means[i] = child.mean()

        sortinds = np.argsort(-child_means)
        child_names = child_names[sortinds]
        child_means = child_means[sortinds]

        num_hotspots = min(num_hotspots, len(self.children))

        for i in range(num_hotspots):
            print(f'{indent}{child_names[i]} - {child_means[i]}s', file=out)
            self.children[i].print(out, level, indent + '  ')

        sum_skipped = child_means[num_hotspots:].sum()
        other_regions = ', '.join(name for name in child_names[num_hotspots:])
        print(f'{indent}Report skipped {sum_skipped}s in regions {other_regions}', file=out)

class LoggingPerfStats(PerfStats):
    """A simple extension to perf stats that logs to a given file regularly

    Attributes:
        logfile (str): the file we log to
        loghandle (filehandle): the handle to the file we are logging into

        interval (float): the time between logging in seconds
        level (str): the level we log at

        last_logged (float): the last time we logged data
    """

    def __init__(self, identifier: str, logfile: str, interval=30.0, level='trace'):
        super().__init__(identifier)

        self.logfile = logfile
        self.loghandle = None
        self.interval = interval
        self.level = level

        self.last_logged = None

    def __del__(self):
        self.close()

    def exit(self):
        """Standard exit and checks if its an appropriate time to log.
        """
        if super().exit()[1]:
            if self.last_logged is None:
                time_since = float('inf')
                self.loghandle = open(self.logfile, 'a')
            else:
                time_since = time.time() - self.last_logged

            if time_since > self.interval:
                self._log()

    def _log(self):
        self.last_logged = time.time()
        ll_pretty = datetime.fromtimestamp(self.last_logged)
        print(f'---Performance as of {ll_pretty}---')
        self.print(out=self.loghandle, level=self.level)
        self.last_logged = time.time() # in case printing takes a while

    def close(self):
        """Closes the open loghandle if there is one"""
        if self.loghandle is not None:
            self.loghandle.close()
            self.loghandle = None
