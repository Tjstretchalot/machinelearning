"""This module contains code for evolving neural networks. This is like a random search -
at each evolution step, we choose a parameter and a few seed points. We then iterate through
the seed points and manually compute the gradient and walk it until we flatten. At the end
of this process we choose the maximum of the best point from all the seeding points.

To prevent going backwards, one of the seeding points is always our current value."""

import shared.setup_torch # pylint: disable=unused-import

import typing
import numpy as np
import scipy.spatial.distance as sdist

from multiprocessing import Queue, Process
from queue import Empty
import importlib
import time
import logging
import logging.handlers
import os
import signal
import sys
import matplotlib.pyplot as plt
import json
from itertools import chain
import shutil
import torch

class EvolvableParam:
    """Describes a single evolvable parameter.

    Attributes:
        name (str): the name of this parameter
        categorical (bool): if this parameter is categorical. A categorical variable takes
            a set of values which cannot be related to each other non-arbitrarily, so it
            is not possible to compute gradients.
    """

    def __init__(self, name: str, categorical: bool):
        self.name = name
        self.categorical = categorical

class NumericEvolvableParam(EvolvableParam):
    """Describes a single numeric evolvable parameter. There are two types of numerical parameters
    - integral and continuous.

    Attributes:
        integral (bool): if true the numbers used are always integers, otherwise the numbers are
            floats.
        domain (tuple[number, number]): the minimum and maximum (inclusive for both) values that
            this parameter can take.
    """

    def __init__(self, name: str, integral: bool,
                 domain: typing.Tuple[typing.Union[int, float], typing.Union[int, float]]):
        super().__init__(name, False)
        self.integral = integral

        assert isinstance(domain, tuple)
        if integral:
            assert isinstance(domain[0], int)
            assert isinstance(domain[1], int)
        else:
            assert isinstance(domain[0], float)
            assert isinstance(domain[1], float)
        self.domain = domain

    @property
    def continuous(self):
        """not integral"""
        return not self.integral

    def get_seeds(self, start: float, num: int) -> np.ndarray:
        """Gets a spread set of starting values within the domain of this parameter.

        Arguments:
            start (float): the current value of the parameter. will be one of the seed points
            num (int): the number of seed points to generate. we will return fewer if it's not
                possible

        Returns:
            A numpy array with dimensions [num] containing the values
        """

        if self.integral:
            if self.domain[1] - self.domain[0] + 1 <= num:
                return np.arange(self.domain[0], self.domain[1] + 1)
            result = np.random.choice(
                np.arange(self.domain[0], self.domain[1] + 1), num, replace=False)
            if start not in result:
                result[0] = start
            return result


        result = np.zeros((num,), dtype='float64')
        result[0] = start
        min_sep = (self.domain[1] - self.domain[0]) * min(0.05, 0.25 / num)

        for i in range(1, num):
            rejections = 0
            while True:
                _pt = np.random.uniform(self.domain[0], self.domain[1], 1)
                if sdist.cdist(_pt.reshape(1, 1), result[:i].reshape(i, 1)).min() < min_sep:
                    rejections += 1
                    if rejections > 50000:
                        raise ValueError(
                            f'failed to sample! domain might be singleton: {self.domain}')
                else:
                    result[i] = _pt
                    break

        return result

    def serialize(self) -> typing.Tuple[str, bool, typing.Tuple[float, float]]:
        """Serializes this parameter into primitives"""
        return (self.name, self.integral, self.domain)

    @classmethod
    def deserialize(cls, serd: typing.Tuple[str, bool, typing.Tuple[float, float]]):
        """Deserializes the result from serialize"""
        return cls(*serd)

class SalienceComputer:
    """Describes something that is able to manually compute salient points to check from
    arbitrary data in order to find the maximum of a given metric.
    """

    @staticmethod
    def hint(domain: typing.Tuple[typing.Union[int, float], typing.Union[int, float]],
             checked: np.ndarray, metric: np.ndarray) -> typing.Union[int, float]:
        """Selects the next best choice for point to select given the known points. This
        should be called after all the seed points have been checked to find the most
        salient points to check.

        WARNING:
            The metric is assumed to be strictly positive. That means that metric values of
            0 are ignored. This is how the calling function may use a variable number of
            trials per point. The 0s should be contiguous and after the real trials.

        Arguments:
            checked (np.ndarray): must be in ascending order
            metric (np.ndarray):
                checked.shape = (num_points,)
                metric.shape = (num_points, num_trials_per_point)

                for a given index 0 <= i < num_points, checked[i] tells us the value for this
                parameter and metric[i] tells us the value of the relevant metric.

                if self.integral then checked dtype should be int32, otherwise checked.dtype
                should be float64. The metric should always be float64, where each value
                corresponds to the value of the metric for that trial.
        Returns:
            next_point (int or float or None):
                if this is integral and we have exhausted the search space, this returns None.
                if this is integral and there are values remaining, returns the most salient value
                if this is not integral, always returns the most salient value to check
        """
        raise NotImplementedError()

class IntegralSalienceComputer(SalienceComputer):
    """Describes a salience computer which works on integers
    """

    @staticmethod
    def hint(domain: typing.Tuple[int, int], checked: np.ndarray, metric: np.ndarray) -> int:
        num_pts = checked.shape[0]
        if num_pts < 3:
            raise Exception('this must have at least 3 points to have any hope')
        assert checked.dtype == 'int32'
        assert metric.dtype == 'float64'
        assert isinstance(domain, tuple)
        assert isinstance(domain[0], int)
        assert isinstance(domain[1], int)

        domain_width = domain[1] - domain[0]
        assert isinstance(domain_width, int)

        if num_pts == domain_width + 1:
            return None

        metric_bests = metric.max(1)
        # this will naturally ignore 0s as we must for variable num trials

        best_ind = int(metric_bests.argmax())
        best = int(checked[best_ind])

        if best_ind == 0:
            if best == domain[0]:
                return IntegralSalienceComputer._hint_near(domain, checked, domain[0], True)

            start = min(best - 1, round(best - domain_width * 0.05))
            return IntegralSalienceComputer._hint_near(domain, checked, start, True)

        if best_ind == num_pts - 1:
            if best == domain[1]:
                return IntegralSalienceComputer._hint_near(domain, checked, domain[1], False)

            start = max(best + 1, round(best + domain_width * 0.05))
            return IntegralSalienceComputer._hint_near(domain, checked, start, False)

        grads = np.gradient(metric_bests)
        # this is a little weird since theres a time distortion implicitly,
        # but it's reasonable enough

        local_grad = -grads[best_ind] / np.sum(np.square(grads))
        # notice the negative since we want maximums

        while abs(local_grad) < 1e-06 or abs(local_grad) > 1 - 1e-06:
            # this shouldn't happen but we shouldn't crash if it does, just wiggle randomly
            local_grad = np.random.uniform(-1, 1)

        # weighted sum. -0.25 means take 25% of the previous and 75% of the current
        if local_grad < 0:
            start = int(round(
                checked[best_ind - 1] * (-local_grad) + checked[best_ind] * (1 + local_grad)))
            return IntegralSalienceComputer._hint_near(domain, checked, start, True)
        start = int(round(
            checked[best_ind + 1] * local_grad + checked[best_ind] * (1 - local_grad)))
        return IntegralSalienceComputer._hint_near(domain, checked, start, False)

    @staticmethod
    def _hint_near(domain: typing.Tuple[int, int],
                   checked: np.ndarray, start: int, fail_from_left: bool):
        assert isinstance(start, int), f'start={start}, type={type(start)} - infected by numpy most likely'
        assert isinstance(domain[0], int)
        assert isinstance(domain[1], int)

        start = min(max(start, domain[0]), domain[1])

        for i in range(min(start - domain[0], domain[1] - start) + 1):
            if start - i not in checked:
                return start - i
            if start + i not in checked:
                return start + i
        if fail_from_left:
            for i in range(domain[0], domain[1] + 1):
                if i not in checked:
                    return i
        else:
            for i in range(domain[1], domain[0] - 1, -1):
                if i not in checked:
                    return i
        raise Exception('wont get here')

class ContinuousSalienceComputer(SalienceComputer):
    """Describes a salience computer which works on a continuous domain
    """

    TARGET_SEP = 0.05

    @staticmethod
    def hint(domain: typing.Tuple[float, float], checked: np.ndarray, metric: np.ndarray) -> float:
        assert checked.dtype == 'float64'
        assert len(checked.shape) == 1
        assert metric.dtype == 'float64'

        num_pts = checked.shape[0]
        if num_pts < 3:
            raise Exception('this must have at least 3 points to have any hope')

        metric_bests = metric.max(1)
        # this will naturally ignore 0s as we must for variable num trials

        best_ind = metric_bests.argmax()
        if best_ind == 0:
            return ContinuousSalienceComputer._hint_left(domain, checked, 0)

        if best_ind == num_pts - 1:
            return ContinuousSalienceComputer._hint_right(domain, checked, num_pts - 1)

        grads = np.gradient(metric_bests)
        local_grad = grads[best_ind]

        if local_grad < 0:
            return ContinuousSalienceComputer._hint_left(domain, checked, best_ind)
        return ContinuousSalienceComputer._hint_right(domain, checked, best_ind)

    @staticmethod
    def _hint_left(domain: typing.Tuple[float, float], checked: np.ndarray, left_of_ind: int):
        domain_width = domain[1] - domain[0]

        if left_of_ind == 0:
            if checked[0] <= domain[0] + domain_width * ContinuousSalienceComputer.TARGET_SEP:
                return ContinuousSalienceComputer._hint_random(domain, checked)
            return np.random.uniform(
                domain[0], checked[0] - domain_width * ContinuousSalienceComputer.TARGET_SEP)

        if checked[left_of_ind] - checked[left_of_ind - 1] > (
                2 * domain_width * ContinuousSalienceComputer.TARGET_SEP):
            return np.random.uniform(
                checked[left_of_ind - 1] + domain_width * ContinuousSalienceComputer.TARGET_SEP,
                checked[left_of_ind] - domain_width * ContinuousSalienceComputer.TARGET_SEP)

        return ContinuousSalienceComputer._hint_random(domain, checked)

    @staticmethod
    def _hint_right(domain: typing.Tuple[float, float], checked: np.ndarray, right_of_ind: int):
        domain_width = domain[1] - domain[0]
        num_pts = checked.shape[0]

        if right_of_ind == num_pts - 1:
            if checked[num_pts - 1] >= (
                    domain[1] - domain_width * ContinuousSalienceComputer.TARGET_SEP):
                return ContinuousSalienceComputer._hint_random(domain, checked)
            return np.random.uniform(
                checked[num_pts - 1] + domain_width * ContinuousSalienceComputer.TARGET_SEP,
                domain[1])

        if checked[right_of_ind + 1] - checked[right_of_ind] > (
                2 * domain_width * ContinuousSalienceComputer.TARGET_SEP):
            return np.random.uniform(
                checked[right_of_ind] + domain_width * ContinuousSalienceComputer.TARGET_SEP,
                checked[right_of_ind + 1] - domain_width * ContinuousSalienceComputer.TARGET_SEP)

        return ContinuousSalienceComputer._hint_random(domain, checked)

    @staticmethod
    def _hint_random(domain: typing.Tuple[float, float], checked: np.ndarray):
        domain_width = domain[1] - domain[0]
        diffs = np.diff(checked)

        valid_regions = diffs > 2 * ContinuousSalienceComputer.TARGET_SEP * domain_width
        num_valid_regions = valid_regions.sum()
        if num_valid_regions == 0:
            best_region_ind = diffs.argmax()
            return np.random.uniform(checked[best_region_ind], checked[best_region_ind + 1])

        region_choice = np.random.randint(num_valid_regions)
        region_ind = np.searchsorted(np.cumsum(valid_regions), region_choice+1)
        return np.random.uniform(
            checked[region_ind] + domain_width*ContinuousSalienceComputer.TARGET_SEP,
            checked[region_ind + 1] - domain_width*ContinuousSalienceComputer.TARGET_SEP)



class CategoricalEvolvableParam(EvolvableParam):
    """Describes a categorical param. These can only take particular values and it does not
    make sense to compute gradients on them, so random sampling without replacement is how
    they should be evolved.

    Attributes:
        values (list[any]): the values that this parameter can take. Must be primitives
            for serialize() to work
    """

    def __init__(self, name: str, values: typing.List[typing.Any]):
        super().__init__(name, True)
        self.values = values

    def __getitem__(self, i):
        return self.values[i]

    def sample(self, start: typing.Any, num: int) -> typing.List[typing.Any]:
        """Fetches a sample of this parameter with at most num results and without
        repetition.

        Arguments:
            start (optional, any): the value (in values) that must be included in the result. if
                none then ignored
            num (int): the maximum number of results to return

        Returns:
            sample (list[any]): a subset of values with at most num entries
        """

        if num >= len(self.values):
            return self.values.copy()

        if start is None:
            sample_inds = np.random.choice(len(self.values), (num,), replace=False)
            result = []
        else:
            index_of_start = self.values.index(start)
            sample_inds = np.random.choice(len(self.values) - 1, (num - 1,), replace=False)
            sample_inds[sample_inds >= index_of_start] += 1

            result = [start]

        for ind in sample_inds:
            result.append(self.values[ind])
        return result

    def serialize(self):
        """Serializes this into primitives"""
        return (self.name, self.values)

    @classmethod
    def deserialize(cls, serd):
        """Deserializes the result from serialize"""
        return cls(*serd)

class EvolutionProblem:
    """Contains a description of a problem that can be solved using an evolutionary algorithm.
    To make usage less tedious, this is susbcriptable by going through categoricals, then
    integrals, then continuous

    Attributes:
        categoricals (list[CategoricalEvolvableParam]): the categorical parameters to the network
        integrals (list[NumericEvolvableParam]): the integral parameters to the network
        continuous (list[NumericEvolvableParam]): the continuous parameters to the network
    """

    def __init__(
            self,
            categoricals: typing.List[CategoricalEvolvableParam],
            integrals: typing.List[NumericEvolvableParam],
            continuous: typing.List[NumericEvolvableParam]):
        self.categoricals = categoricals
        self.integrals = integrals
        self.continuous = continuous

    def __len__(self):
        return len(self.categoricals) + len(self.integrals) + len(self.continuous)

    def __getitem__(self, i) -> EvolvableParam:
        if i < len(self.categoricals):
            return self.categoricals[i]
        i -= len(self.categoricals)
        if i < len(self.integrals):
            return self.integrals[i]
        i -= len(self.integrals)
        return self.continuous[i]

    def __iter__(self):
        for val in self.categoricals:
            yield val
        for val in self.integrals:
            yield val
        for val in self.continuous:
            yield val

    def realize(self, values: typing.Dict[str, typing.Any], **kwargs) -> typing.Any:
        """Accepts the values for each parameter (categorical, integral, and continuous) and
        converts it into a realized representation (depends on the particular instance). May
        require additional keyword arguments (i.e. input/output dimension for a network)

        Arguments:
            values (dict[str, any]): the values for each of the parameters of the problem
            kwargs (dict): see the particular instance you are using

        Returns:
            realized (any): the realized version of this problem
        """
        raise NotImplementedError()

    def serialize(self):
        """Serializes this problem into primitives"""
        return (tuple(cat.serialize() for cat in self.categoricals),
                tuple(i.serialize() for i in self.integrals),
                tuple(cont.serialize() for cont in self.continuous))

    @classmethod
    def deserialize(cls, serd):
        """Deserializes the result from serialize"""
        return cls(
            [CategoricalEvolvableParam.deserialize(cat) for cat in serd[0]],
            [NumericEvolvableParam.deserialize(i) for i in serd[1]],
            [NumericEvolvableParam.deserialize(cont) for cont in serd[2]])

class SubtypableSolution(CategoricalEvolvableParam):
    """For a particular problem there are often various approaches to solving it which give
    rise to different evolution problems. This is a particular type of categorical parameter
    where the values are either other SubtypableSolution's or EvolutionProblem's. In order
    to actually solve anything, you must drill down to a real EvolutionProblem. This can
    also be treated as a categorical evolvable parameter where the full path corresponds
    to the key and the problems are the output, and the distances are tree-style.

    It is typically more convenient to construct SubtypableSolutions then flatten them for
    actual use. It is also easier to reference this class than describe this every time.
    """

    def __init__(self, name: str,
                 values: typing.List[typing.Union['SubtypableSolution', EvolutionProblem]]):
        super().__init__(name, values)

    def flatten(self) -> CategoricalEvolvableParam:
        """Flattens this problem into a single categorical evolvable parameter where the keys
        are the paths in this tree and the values are the leaf nodes (EvolutionProblem's)
        """
        new_vals = []
        stack = self.values.copy()
        while stack:
            cur = stack.pop()
            if isinstance(cur, EvolutionProblem):
                new_vals.append(cur)
            else:
                for val in cur.values:
                    stack.append(val)
        return CategoricalEvolvableParam(self.name, new_vals)


class CompleteEvolutionProblem:
    """Describes the complete evolution problem required for initiating a network.

    Attributes:
        input_dim (int): the input dimension of the problem to solve
        output_dim (int): the output dimension of the problem to solve

        approaches (CategoricalEvolvableParam): a flattened SubtypableSolution where the
            EvolutionProblems produce GenericTrainers and Networks (a tuple). This should
            not include the special sensitive values, which will be passed in through the kwargs

        sensitives (EvolutionProblem): describes the special training parameters that must be tuned
            to at least some degree between every other evolution in order to get a fair measure
            of their effectiveness. For example, learning rate and batch size. This should realize
            to a dict which is passed as kwargs to the approach that is being used.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 approaches: CategoricalEvolvableParam, sensitives: EvolutionProblem):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.approaches = approaches
        self.sensitives = sensitives

# This section is now for actually solving evolution problems

class SensitiveSweepInfo(typing.NamedTuple):
    """When evolving we sweep each sensitive parameter whenever we want to tweak a real parameter,
    then we take the best set of those (whatever we sweeped last) as the representative value. This
    is telling us about the result of sweeping a particular sensitive parameter and holding
    everything else fixed.

    Attributes:
        param (EvolvableParam): the sensitive parameter that we swept
        checked (list[num_points]): the values that we checked
        metric (ndarray[num_points, num_trials_per_point]): the value of the metric for each trial
            at each point
        selected (any): the value for the sensitive parameter that we chose
        selected_ind (int): the index in checked / metric that we chose
    """
    param: EvolvableParam
    checked: typing.List[typing.Any]
    metric: np.ndarray
    selected: typing.Any
    selected_ind: int

class ParamSweepPointInfo(typing.NamedTuple):
    """Describes the intrasweep result of trying to change the particular param to the given
    value. We do this by fixing the param to the specified value, sweeping over the sensitive
    variables (potentially multiple times to account for higher-order effects), then selecting
    the last value in each sweep.

    Attributes:
        param (EvolvableParam): the parameter we changed
        value (any): the value we set the parameter to
        sens_sweeps (list[SensitiveSweepInfo]) the sweeps we performed in the order we performed
            them
        selected (dict[EvolvableParam, any]): the values we chose for each of the sensitive
            parameters
    """
    param: EvolvableParam
    value: typing.Any
    sens_sweeps: typing.List[SensitiveSweepInfo]
    selected: typing.Dict[EvolvableParam, typing.Any]

def work(workerid: int, problem_import: str, problem_func: str, savepath: str,
         jobqueue: Queue, resultqueue: Queue):
    """Keeps doing the work in the jobqueue and pushing it into the resultqueue until
    we get the finished message.

    Arguments:
        workerid (int): the identifier for this worker. used to generate the logger
            for the trainer
        problem_import (str): a module name, i.e. 'gaussian_spheres.evolve_runner'
        problem_func (str): a function in the problem_import module, i.e. 'create_problem'
        jobqueue (Queue): the queue which we will receive jobs from. A job is
            (approach_index: int, values: dict[str, any], special_vals: dict[str, any]).
            the values keys are interpeted as the name of the parameter.
            Sending 'False' through the jobqueue will shutdown the worker once the result queue is
            empty.
        resultqueue (Queue): the queue we will send jobs to. results from jobs are just the
            result from the trainer, which is a dict.
    """
    import random
    seedmax = 2**30
    for _ in range(100000):
        random.randrange(seedmax)
    random.seed(time.time())
    for _ in range(2000):
        random.randrange(seedmax)

    seed1 = random.randrange(seedmax)
    seed2 = random.randrange(seedmax)
    np.random.seed(seed1)
    torch.manual_seed(seed2)


    mod = importlib.import_module(problem_import)
    problem: CompleteEvolutionProblem = getattr(mod, problem_func)()

    logger = logging.getLogger(f'{__name__}_worker_{workerid}')
    logger.setLevel(logging.DEBUG)

    handler = logging.handlers.RotatingFileHandler(
        os.path.join(savepath, f'worker_{workerid}.log'),
        maxBytes=1024*256, backupCount=5)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p'))
    logger.addHandler(handler)


    signal.signal(signal.SIGINT, lambda *args: logger.critical('Ignoring SIGINT'))

    try:
        logger.info('Worker spawned: numpy seed=%s, torch seed=%s', seed1, seed2)
        while True:
            while True:
                try:
                    job = jobqueue.get()
                    break
                except InterruptedError:
                    logger.exception('jobqueue.get() interrupted - ignoring')

            if not job:
                logger.info('got shutdown message')
                break

            logger.info('got new job: %s', str(job))
            approach: EvolutionProblem = problem.approaches[job[0]]
            trainer, network = approach.realize(job[1], **job[2])
            result = trainer.train(network, logger=logger)
            #result = {'accuracy': np.random.uniform(0, 1) + np.sin(job[2]['learning_rate'] * ((2*np.pi) / 0.1))} # for testing pylint: disable=line-too-long
            logger.info('finished job with result: %s', str(result))
            resultqueue.put(result)
    except: #pylint: disable=bare-except
        logger.exception('Fatal exception')

    while not resultqueue.empty():
        time.sleep(0.1)
        while not jobqueue.empty():
            try:
                jobqueue.get_nowait()
            except Empty:
                pass

    while not jobqueue.empty():
        try:
            jobqueue.get_nowait()
        except Empty:
            pass

    logger.info('shutdown')
    logging.shutdown()


class EvolverSettings(typing.NamedTuple):
    """This contains all the initial settings that are passed to the evolver.

    Attributes:
        problem_import (str): the module in which the problem lives
        problem_func (str): the name of the function that returns the problem
        save_path (str): the folder where we can save run information
        duration_ms (int): how long to sweep for
        cores (int): the number of cores we are meant to use
        metric_name (str): the name of the metric (the key in the result from the trainer produced
            by the problem) that we are using
        max_trials (int): the maximum number of trials we should do for a single point in a
            sweep
        trial_patience (int): the number of trials without a change we should do at a single point
        trial_epsilon (float): the minimum improvement for the metric to reset our patience when
            trying to determine the networks performance in a given setting
        seed_points (int): the number of points we will seed each sweep with. for categorical
            parameters, this is the total number of categories tttempted
        salient_points (int): for numeric parameters, this is the number of poitns we pick in a
            clever manner after seeding during a parameter sweep
        evolve_patience (int): the number of generations we are willing to go without improvement
        evolve_epsilon (float): the minimum change in the metric to reset the evolution patience
        pass_patience (int): the number of passes across the sensitive variables we will do without
            improvement before we are happy with the metrics value for the approach parameter.
            Higher values are more likely to detect higher-order effects between the sensitive
            parameters
        max_passes (int): the maximum number of passes before terminating
    """
    problem_import: str
    problem_func: str
    save_path: str
    duration_ms: int
    cores: int
    metric_name: str
    max_trials: int
    trial_patience: int
    trial_epsilon: float
    seed_points: int
    salient_points: int
    evolve_patience: int
    evolve_epsilon: float
    pass_patience: int
    max_passes: int

class EvolverWorker(typing.NamedTuple):
    """This contains the main threads description of a worker, which is a process that is running
    work()

    Attributes:
        process (Process): the actual process
        job_queue (Queue): the queue we can send (approach_index, approach_params, sens_params) to
        result_queue (Queue): the queue the can receive the result from the trainer
    """
    process: Process
    job_queue: Queue
    result_queue: Queue

class EvolutionContext:
    """Contains information about an evolution that is currently occuring. This has all the major
    code to evolve, just requires some glue.

    Attributes:
        evolve_param (EvolvableParam): the parameter that we are trying to evolve
        checked (list[ParamSweepPointInfo]): the points that we have already checked

        seed_points (mapping[int, any]): the seed points that we got from the parameter
        seed_points_ind (int): the index in seed_points we are currently working on, or
            len(seed_points) if we're done with the seed points

        salient_points_ind (optional, int): if seed_points_ind == len(seed_points), then this
            corresponds the index of the salient point we are currently working on. when it reaches
            the settings # salient points, we are done. this is ignored if the parameter is
            categorical

        sweep_len (int): the number of points in sweep. If the evolve parameter is categorical,
            this is just len(sweep). Otherwise, it means that only the first sweep_len points are
            real and the rest are padding.
        sweep_pts (optional[np.ndarray]): if the evolve parameter is numerical then this is
            ndarray(settings.seed_points + settings.salient_points) where only the first
            sweep_len values are real and the rest are padding, and it is maintained in
            sorted order.
        sweep (union[np.ndarray, dict]): if the evolve parameter is numerical, this is a
            ndarray(settings.seed_points + settings.salient_points). if the
            sensitive parameter is categorical, this is a dict[parameter value, float].
            For the evolve parameter we do not have multiple trials because that would be
            redundant (we are performing multiple passes per point)

        new_sensitives (dict[str, any]): the sensitive values that we are currently using

        checking (typing.Any): the value for the parameter that we are currently attempting
        pass_ (int): the number of times we have gone through the sensitive parameters
        patience_used (int): the number of passes we have gone through without improvement
        best (float): the best value for the metric we have seen at the checking value
        improved (bool): if we have improved the metric this pass
        sensitive (tuple[int, EvolvableParam]): the index in the approach sensitives and the
            actual value of the parameter that we are currently sweeping
        sens_seed_points (mapping[int, any]): the seed points for the sensitive parameter
        sens_seed_points_ind (int): the index in sens_seed_points that we are currently on
        sens_salient_points_ind (optional, int): if the sensitive parameter is numerical, then
            this corresponds to the salient point index we are currently working on. this is
            ignored if the sensitive parameter is categorical.
        sens_checking (any): the value for the sensitive we are currently checking
        sens_sweep_len (int): the number of points in sens_sweep. if the sensitive
            parameter is categorical, this is just len(sens_sweep). Otherwise, it means that
            only the first sens_sweep_len values are real
        sens_sweep_pts (optional[np.ndarray]): if the sensitive parameter is numerical then
            this is ndarray(settings.seed_points + settings.salient_points) where only the first
            sens_sweep_len values are real and the rest a
        sens_sweep (union[np.ndarray, dict]): if the sensitive parameter is numerical, this is
            ndarray(settings.seed_points + settings.salient_points, settings.max_trials). so for
            a given index 0 <= i < sens_sweep_len, sens_sweep_pts[i] is the point and for
            0 <= j < max_trials, sens_sweep[i][j] is either a positive value corresponding
            to the metrics value on trial j or 0 if that trial was not performed..

            if the sensitive parameter is categorical, this is a dict where the keys are the values
            for the parameter and the values are ndarray(settings.max_trials) in the same way as
            before.
        metric_through_passes (ndarray): shape [max_passes x number sensitives] - stores a history
            of the accuracy
        changes_through_passes (list[any]): shape [max_passes x number_sensitives] - stores a history
            of what we changed each pass. the value is the new value for the sensitive parameter
    """
    def __init__(self, evolve_param: EvolvableParam):
        self.evolve_param = evolve_param
        self.checked: typing.List[ParamSweepPointInfo] = []
        self.seed_points: typing.Mapping[int, typing.Any] = None
        self.seed_points_ind: int = None
        self.salient_points_ind: int = None
        self.sweep_pts: typing.Optional[np.ndarray] = None
        self.sweep: typing.Union[np.ndarray, typing.Dict[typing.Any, np.ndarray]] = None
        self.sweep_len: int = None
        self.new_sensitives: typing.Dict[str, typing.Any] = None

        self.checking = None
        self.pass_: int = None
        self.patience_used: int = None
        self.best: float = None
        self.improved: bool = None
        self.sensitive: typing.Tuple[int, EvolvableParam] = None
        self.sens_seed_points: typing.Mapping[int, typing.Any] = None
        self.sens_seed_points_ind: int = None
        self.sens_salient_points_ind: typing.Optional[int] = None
        self.sens_checking: typing.Any = None
        self.sens_sweep_len: int = None
        self.sens_sweep_pts: typing.Optional[np.ndarray] = None
        self.sens_sweep: typing.Union[np.ndarray, typing.Dict[typing.Any, float]] = None
        self.metric_through_passes: np.ndarray = None
        self.changes_through_passes: typing.List[typing.Any] = None

    def seed(self, evolver: 'Evolver'):
        """Initializes the evolve parameter sweep with the appropriate seed points. From here
        start_checking can be called
        """
        settings = evolver.settings

        self.seed_points_ind = 0
        self.sweep_len = 0
        if self.evolve_param.categorical:
            self.seed_points = self.evolve_param.sample(
                evolver.approach_params[self.evolve_param.name], settings.seed_points)
            self.sweep_pts = None
            self.sweep = dict()
        else:
            self.seed_points = self.evolve_param.get_seeds(
                evolver.approach_params[self.evolve_param.name], settings.seed_points)
            self.sweep_pts = np.zeros(
                settings.seed_points + settings.salient_points, dtype=self.seed_points.dtype)
            self.sweep = np.zeros(self.sweep_pts.shape[0], dtype='float64')

        self.new_sensitives = evolver.sensitive_params.copy()

    def choose_check(self, evolver: 'Evolver') -> bool:
        """Decides the next value to check for the evolve_param. If there are no more points to
        check, returns False and you should call finish(). If this returns True, move onto
        increment_sensitive"""

        if self.checking is None:
            # first!
            self._start_checking(evolver, self.seed_points[0])
            return True

        if self.seed_points_ind < len(self.seed_points):
            self.seed_points_ind += 1
            if self.seed_points_ind < len(self.seed_points):
                self._start_checking(evolver, self.seed_points[self.seed_points_ind])
                return True
            if self.evolve_param.categorical:
                return False

            self.salient_points_ind = 0
            if evolver.settings.salient_points == 0:
                return False
        else:
            self.salient_points_ind += 1
            if self.salient_points_ind >= evolver.settings.salient_points:
                return False

        if self.evolve_param.integral:
            salient_pt = IntegralSalienceComputer.hint(
                self.evolve_param.domain, self.sweep_pts[:self.sweep_len],
                self.sweep[:self.sweep_len].reshape(self.sweep_len, 1))
            if salient_pt is None:
                return False
        else:
            salient_pt = ContinuousSalienceComputer.hint(
                self.evolve_param.domain, self.sweep_pts[:self.sweep_len],
                self.sweep[:self.sweep_len].reshape(self.sweep_len, 1))

        self._start_checking(evolver, salient_pt)
        return True

    def _start_checking(self, evolver: 'Evolver', value: typing.Any):
        """Prepares to find the sensitive parameters that work best when the evolve parameter is
        set to the specified value. From here we must repeatedly iterate through the sensitive
        parameters until we exhaust our pass patience."""

        evolver.logger.info('Checking value=%s', str(value))
        self.checking = value
        self.pass_ = 0
        self.patience_used = 0
        self.best = float('-inf')
        self.improved = False
        self.metric_through_passes = (
            np.zeros((evolver.settings.max_passes, len(evolver.problem.sensitives)),
                     dtype='float64'))
        self.changes_through_passes = []
        self.sensitive = None

    def increment_sensitive(self, evolver: 'Evolver') -> bool:
        """Increments the sensitive that we are going to sweep. This will find seeds and prepare
        us for the following loop: perform_trials, select_sensitive_point

        Returns:
            cont (bool): True if we found a new sensitive to sweep, False if we have exhausted
                the pass patience and should stop by calling end_checking
        """

        if not self.sensitive:
            # First!
            self.sensitive = (0, evolver.problem.sensitives[0])
            self.changes_through_passes.append([])
        else:
            new_ind = self.sensitive[0] + 1
            if new_ind == len(evolver.problem.sensitives):
                self.pass_ += 1 # if we terminate after the first, we want pass_ to be 1
                if not self.improved:
                    self.patience_used += 1
                    if self.patience_used >= evolver.settings.pass_patience:
                        return False
                else:
                    self.improved = False
                    self.patience_used = 0
                new_ind = 0
                self.changes_through_passes.append([])
            self.sensitive = (new_ind, evolver.problem.sensitives[new_ind])

        self.sens_seed_points_ind = 0
        if self.sensitive[1].categorical:
            self.sens_seed_points = self.sensitive[1].sample(
                self.new_sensitives[self.sensitive[1].name], evolver.settings.seed_points)
            self.sens_checking = self.sens_seed_points[0]
            self.sens_sweep_len = 0
            self.sens_sweep_pts = None
            self.sens_sweep = dict()
            self.sens_sweep[self.sens_checking] = (
                np.zeros(evolver.settings.max_trials, dtype='float64'))
        else:
            self.sens_seed_points = self.sensitive[1].get_seeds(
                self.new_sensitives[self.sensitive[1].name], evolver.settings.seed_points)

            expected_dtype = 'int32' if self.sensitive[1].integral else 'float64'
            if self.sens_seed_points.dtype != expected_dtype:
                raise Exception(f'strange seed dtype: got {self.sens_seed_points.dtype}, expected {expected_dtype}')

            self.sens_checking = self.sens_seed_points[0]
            self.sens_sweep_len = 0
            self.sens_sweep_pts = np.zeros(
                evolver.settings.seed_points + evolver.settings.salient_points,
                dtype=self.sens_seed_points.dtype)
            self.sens_sweep = np.zeros(
                (self.sens_sweep_pts.shape[0], evolver.settings.max_trials), dtype='float64')

        evolver.logger.info('Sweeping sensitive=%s (starting with val=%s)', self.sensitive[1].name, self.sens_checking)
        return True

    def perform_trials(self, evolver: 'Evolver'):
        """Performs all the necessary trials for the current sensitive and stores the result
        in the state. This should be followed by a call to select_sensitive_point.

        This populates the sens_sweep with the trial results using the evolvers workers
        """

        approach_ind = evolver.approach[0]

        approach_params = evolver.approach_params.copy()
        approach_params[self.evolve_param.name] = self.checking

        sens_params = self.new_sensitives.copy()
        sens_params[self.sensitive[1].name] = self.sens_checking

        trial_best = float('-inf')
        trial_patience = evolver.settings.trial_patience
        trial_epsilon = evolver.settings.trial_epsilon
        trial_patience_used = 0
        trial_index = 0

        if self.sensitive[1].categorical:
            metric_store = self.sens_sweep[self.sens_checking]
        else:
            evolver.logger.debug('sens_sweep_pts=%s, sens_sweep_len=%s, sens_checking=%s', self.sens_sweep_pts, self.sens_sweep_len, self.sens_checking)
            insert_ind = (
                np.searchsorted(self.sens_sweep_pts[:self.sens_sweep_len], self.sens_checking)
                if self.sens_sweep_len > 0
                else 0
            )
            assert isinstance(insert_ind, (int, np.int32, np.int64)), f'insert_ind={insert_ind}, type(insert_ind)={type(insert_ind)}'
            if insert_ind < self.sens_sweep_len:
                self.sens_sweep_pts[insert_ind+1:self.sens_sweep_len+1] = (
                    self.sens_sweep_pts[insert_ind:self.sens_sweep_len])
                self.sens_sweep_pts[insert_ind] = self.sens_checking

                self.sens_sweep[insert_ind+1:self.sens_sweep_len+1] = (
                    self.sens_sweep[insert_ind:self.sens_sweep_len])
                self.sens_sweep[insert_ind, :] = 0
            else:
                self.sens_sweep_pts[insert_ind] = self.sens_checking
            metric_store = self.sens_sweep[insert_ind]

        while (trial_index < evolver.settings.max_trials
               and trial_patience_used < trial_patience):
            for worker in evolver.workers:
                worker.job_queue.put((approach_ind, approach_params.copy(), sens_params.copy()))
            evolver.logger.debug('dispatched jobs')

            for worker in evolver.workers:
                while True:
                    try:
                        result = worker.result_queue.get()
                        break
                    except InterruptedError:
                        evolver.logger.critical('result_queue.get() was interrupted')

                if trial_index == evolver.settings.max_trials:
                    continue
                result_metric = result[evolver.settings.metric_name]
                metric_store[trial_index] = result_metric
                trial_index += 1

                if result_metric - trial_epsilon > trial_best:
                    evolver.logger.debug('got trial metric %s (improved old: %s)', result_metric, trial_best)
                    trial_best = result_metric
                    if trial_patience_used < trial_patience:
                        trial_patience_used = 0
                elif trial_patience_used < trial_patience:
                    trial_patience_used += 1
                    evolver.logger.debug('got trial metric %s, exhausted patience %s/%s',
                                         result_metric, trial_patience_used, trial_patience)
                else:
                    evolver.logger.debug('got trial metric %s (worse, but already out of patience)', result_metric)

    def select_sensitive_point(self, evolver: 'Evolver') -> bool:
        """Attempts to select the next sensitive point for the sweep. This should be called
        after the first perform_trials and if it returns True should be followed by
        perform_trials. Otherwise, call finish_sensitive then increment_sensitive

        Works as follows:

        Seed point to do?
            yes -> set this to do that seed point, return True
            no  -> categorical?
                yes -> return False
                no  -> more salient points to do?
                    yes -> set this to that salient point, return True
                    no  -> return False
        """

        self.sens_sweep_len += 1

        if self.sensitive[1].categorical:
            current_bests = dict((k, v.max()) for k,v in self.sens_sweep.items())
            evolver.logger.debug('Sensitive %s bests: %s', self.sensitive[1].name, current_bests)
            del current_bests
        else:
            evolver.logger.debug('Sensitive %s | keys=%s, bests=%s', self.sensitive[1].name,
                                 self.sens_sweep_pts[:self.sens_sweep_len],
                                 self.sens_sweep[:self.sens_sweep_len].max(1))

        if self.sens_seed_points_ind < len(self.sens_seed_points):
            self.sens_seed_points_ind += 1
            if self.sens_seed_points_ind < len(self.sens_seed_points):
                self.sens_checking = self.sens_seed_points[self.sens_seed_points_ind]
                evolver.logger.info('Sweeping seed %s/%s: %s', self.sens_seed_points_ind+1,
                                    len(self.sens_seed_points), self.sens_checking)
                if self.sensitive[1].categorical:
                    self.sens_sweep[self.sens_checking] = (
                        np.zeros(evolver.settings.max_trials, dtype='float64'))
                return True

            if self.sensitive[1].categorical:
                evolver.logger.info('Categorical sensitive and no more seed points -> done')
                return False

            self.sens_salient_points_ind = 0
            if evolver.settings.salient_points == 0:
                return False
        else:
            self.sens_salient_points_ind += 1
            if self.sens_salient_points_ind == evolver.settings.salient_points:
                return False

        if self.sensitive[1].integral:
            assert self.sens_sweep_pts.dtype == 'int32'
            salient_pt = IntegralSalienceComputer.hint(
                self.sensitive[1].domain, self.sens_sweep_pts[:self.sens_sweep_len],
                self.sens_sweep[:self.sens_sweep_len])
            if salient_pt is None:
                return False
            self.sens_checking = salient_pt
        else:
            assert self.sens_sweep_pts.dtype == 'float64'
            self.sens_checking = ContinuousSalienceComputer.hint(
                self.sensitive[1].domain, self.sens_sweep_pts[:self.sens_sweep_len],
                self.sens_sweep[:self.sens_sweep_len])

        evolver.logger.info('Sweeping salient %s/%s: %s', self.sens_salient_points_ind+1,
                            evolver.settings.salient_points, self.sens_checking)
        return True

    def finish_sensitive(self, evolver: 'Evolver'):
        """Should be called after select_sensitive_point returns false. This finds the best value
        for the sensitive we just swept, puts it to new_sensitives, and checks if we just improved
        our metric for this pass. It also plots an stores the sensitive sweep."""

        file_name_fmt = f'check_{self.checking}_pass_{self.pass_}_{self.sensitive[1].name}'
        plot_path = os.path.join(evolver.settings.save_path, 'current', file_name_fmt + '.png')
        raw_data_path = os.path.join(evolver.settings.save_path, 'current', file_name_fmt + '.npz')

        if os.path.exists(plot_path) or os.path.exists(raw_data_path):
            raise FileExistsError() # almost certainly a programming error

        if self.sensitive[1].categorical:
            sens_best_val: int = None
            sens_best_metric: float = float('-inf')
            sens_raw_metric = np.zeros((len(self.sens_sweep), evolver.settings.max_trials),
                                       dtype='float64')
            sens_raw_metric_coords = np.zeros_like(sens_raw_metric)
            sens_raw_metric_labels = []
            counter = 0
            for sens_val, sens_metric in self.sens_sweep.items():
                sens_raw_metric[counter, :] = sens_metric
                sens_raw_metric_coords[counter, :] = counter + 1
                sens_raw_metric_labels.append(str(sens_val))
                counter += 1

                sens_metric_scalar = sens_metric.max()
                if sens_metric_scalar > sens_best_metric:
                    sens_best_val = sens_val
                    sens_best_metric = sens_metric_scalar
        else:
            sens_raw_metric = self.sens_sweep[:self.sens_sweep_len].copy()
            sens_raw_metric_coords = (self.sens_sweep_pts[:self.sens_sweep_len].reshape(1, -1)
                                      .T
                                      .repeat(evolver.settings.max_trials, axis=1))
            clear_vals = sens_raw_metric == 0
            sens_raw_metric[clear_vals] = np.NaN
            sens_raw_metric_labels = [f'{v:.2e}' if isinstance(v, (float, np.float32, np.float64)) else str(v) for v in self.sens_sweep_pts[:self.sens_sweep_len]]
            sens_best_ind = np.nanargmax(np.nanmax(sens_raw_metric, 1))
            sens_best_val = self.sens_sweep_pts[sens_best_ind]
            sens_best_metric = self.sens_sweep[sens_best_ind].max()

        evolver.logger.info('Finished sweeping %s - best: %s (%s: %s)', self.sensitive[1].name,
                            sens_best_val, evolver.settings.metric_name, sens_best_metric)

        _, ax = plt.subplots()
        ax.set_title(f'Evolve {self.evolve_param.name} - {self.sensitive[1].name} - {self.pass_ + 1}')
        ax.set_xlabel(self.sensitive[1].name)
        ax.set_ylabel(evolver.settings.metric_name)
        ax.scatter(sens_raw_metric_coords, sens_raw_metric, 1, '#000000', alpha=0.8)
        ax.plot(sens_raw_metric_coords[:, 0], np.nanmax(sens_raw_metric, 1), alpha=0.5, color='#1465e8')
        ax.plot(sens_raw_metric_coords[:, 0], np.nanmean(sens_raw_metric, 1), alpha=0.5, color='#000000')
        ax.hlines(sens_best_metric, sens_raw_metric_coords[:, 0].min(),
                  sens_raw_metric_coords[:, 0].max(), colors='#000000',
                  linestyles='dashed', alpha=0.5)
        ax.set_xticks(sens_raw_metric_coords[:, 0])
        ax.set_xticklabels(sens_raw_metric_labels, fontsize='xx-small')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        np.savez_compressed(raw_data_path, metric=sens_raw_metric,
                            metric_coords=sens_raw_metric_coords,
                            labels=np.array(sens_raw_metric_labels))

        self.metric_through_passes[self.pass_, self.sensitive[0]] = sens_best_metric
        self.changes_through_passes[self.pass_].append(sens_best_val)
        self.new_sensitives[self.sensitive[1].name] = sens_best_val
        if sens_best_metric > self.best:
            self.best = sens_best_metric
            self.improved = True

    def end_checking(self, evolver: 'Evolver'):
        """Should be called after increment_sensitive returns False. This will increment sweep_len
        and populate sweep_pts and sweep. From here you should call choose_check.

        This will also plot and store the accuracy through passes
        """

        file_name_fmt = f'check_{self.checking}_summary'
        plot_path = os.path.join(evolver.settings.save_path, 'current', f'{file_name_fmt}.png')
        raw_data_path = os.path.join(evolver.settings.save_path, 'current', f'{file_name_fmt}.npz')

        if os.path.exists(plot_path) or os.path.exists(raw_data_path):
            raise FileExistsError() # almost certainly a programming error

        num_sens = len(evolver.problem.sensitives)

        _, ax = plt.subplots()
        _checking = (
            self.checking
            if not isinstance(self.checking, (int, float))
            else f'{self.checking:.4f}'
        )
        ax.set_title(f'Evolve {self.evolve_param.name} - {_checking}'
                     + f' - {evolver.settings.metric_name} through time')
        ax.set_xlabel('# Sensitives Redone')
        ax.set_ylabel(evolver.settings.metric_name)

        if self.pass_ > 1:
            ax.vlines((np.arange(self.pass_ - 1) * num_sens) + num_sens, 0, 1,
                      colors='#000000', linestyles='dashed', alpha=0.5)

        flattened_metric = self.metric_through_passes[:self.pass_].flatten()
        indiced = np.arange(flattened_metric.shape[0]) + 1

        flattened_formatted_changes = []
        for pass_ind in range(self.pass_):
            for sens_ind, sens in enumerate(evolver.problem.sensitives):
                sens_val_chosen = self.changes_through_passes[pass_ind][sens_ind]
                flattened_formatted_changes.append(f'{sens.name}\n{sens_val_chosen:.4f}')

        ax.plot(indiced, flattened_metric, alpha=0.8, color='#1465e8')

        ax.set_xticks(indiced)
        ax.set_xticklabels(flattened_formatted_changes, fontsize='x-small', rotation='vertical')

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        np.savez_compressed(
            raw_data_path, metric_through_passes=self.metric_through_passes,
            flattened_formatted_changes=flattened_formatted_changes,
            changes_through_passes=self.changes_through_passes)


        if self.evolve_param.categorical:
            self.sweep[self.checking] = flattened_metric[-1]
        else:
            insert_ind = (
                np.searchsorted(self.sweep_pts[:self.sweep_len], self.checking)
                if self.sweep_len > 0
                else 0
            )

            self.sweep_pts[insert_ind+1:self.sweep_len+1] = (
                self.sweep_pts[insert_ind:self.sweep_len])
            self.sweep_pts[insert_ind] = self.checking
            self.sweep[insert_ind+1:self.sweep_len+1] = self.sweep[insert_ind:self.sweep_len]
            self.sweep[insert_ind] = flattened_metric[-1]

        evolver.logger.debug('Value=%s finished (metric: %s)', _checking, str(flattened_metric[-1]))
        self.sweep_len += 1

    def finish(self, evolver: 'Evolver'
              ) -> typing.Tuple[float, typing.Any, typing.Dict[str, typing.Any]]:
        """This should be called after choose_check returns False. When this is invoked, all the
        required information is already computered. This saves the overall sweep plots and saves
        the summary data.

        Returns:
            metric_final (float): the metric for the final configuration
            new_value (any): the best value for the metric that we found
            new_sensitives (dict[str, any]): the new values for the sensitives in the final
                configuration
        """

        file_name_fmt = f'a_evolve_{self.evolve_param.name}_summary' # a_ -> first
        plot_path = os.path.join(evolver.settings.save_path, 'current', f'{file_name_fmt}.png')
        raw_data_path = os.path.join(evolver.settings.save_path, 'current', f'{file_name_fmt}.npz')

        if os.path.exists(plot_path) or os.path.exists(raw_data_path):
            raise FileExistsError() # almost certainly a programming error

        if self.evolve_param.categorical:
            metric_final, new_value = None, None

            pt_coords = np.arange(self.sweep_len) + 1
            metric_vals = np.zeros(self.sweep_len, dtype='float64')
            counter = 0
            pt_labels = []
            for val, metr in self.sweep.items():
                metric_vals[counter] = metr
                pt_labels.append(str(val))
                counter += 1

                if new_value is None or metr > metric_final:
                    metric_final = metr
                    new_value = val
        else:
            best_ind = self.sweep[:self.sweep_len].argmax()
            metric_final = self.sweep[best_ind]
            new_value = self.sweep_pts[best_ind]

            pt_coords = self.sweep_pts[:self.sweep_len]
            metric_vals = self.sweep[:self.sweep_len]
            pt_labels = [f'{v:.2e}' if isinstance(v, (float, np.float32, np.float64)) else str(v) for v in pt_coords]

        _, ax = plt.subplots()
        ax.set_title(f'Evolve {self.evolve_param.name}')
        ax.set_xlabel(self.evolve_param.name)
        ax.set_ylabel(evolver.settings.metric_name)
        ax.plot(pt_coords, metric_vals, alpha=0.8, color='#1465e8')
        ax.hlines([metric_vals.min(), metric_vals.max()], pt_coords.min(), pt_coords.max(),
                  colors='#000000', linestyles='dashed', alpha=0.5)
        ax.set_xticks(pt_coords)
        ax.set_xticklabels(pt_labels, fontsize='xx-small', rotation='vertical')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        np.savez_compressed(raw_data_path, pt_coords=pt_coords, metric_vals=metric_vals,
                            pt_labels=np.array(pt_labels))

        return metric_final, new_value, self.new_sensitives


def _ignore_del_dir_failure(func, path, excinfo): # pylint: disable=unused-argument
    if func == os.rmdir: # pylint: disable=comparison-with-callable
        return

    raise excinfo[0].with_traceback(excinfo[1], excinfo[2])

class Evolver:
    """The main runner for this module. Takes in the problem specified in an evolvable way and
    evolves until the time runs out or we run out of patience.

    Attributes:
        settings (EvolverSettings): the initial settings
        start_time (float): the time.time() when we started (seconds since epoch)
        logger (logging.Logger): the logger we can use to print stuff

        problem (CompleteEvolutionProblem): the evolution problem we are trying to solve

        generation (int): the generation we are currently at
        approach (tuple[int, EvolutionProblem]): approach index and value that we are currently
            using to solve the problem
        approach_params (dict[str, any]): the parameters for the approach that we are currently
            using to solve the problem. the keys are the names of the parameters
        sensitive_params (dict[str, any]): the parameters for the sensitive values that we are
            currently using to solve the problem.

        workers (list[EvolverWorker]): the workers that are currently spawned

        evolve_context (EvolutionContext): contains the variables related to the current evolution
            that is occuring, or null if one is not occuring

        stop_requested (bool): if a SIGINT has been recieved
        last_stop_request (float): time.time() for last stop request. getting multiple we in a row
            will kill us ungracefully
    """

    def __init__(self, settings: EvolverSettings):
        self.settings = settings
        self.start_time = time.time()
        self.logger: logging.Logger = None

        mod = importlib.import_module(settings.problem_import)
        self.problem: CompleteEvolutionProblem = getattr(mod, settings.problem_func)()

        self.generation: int = None
        self.approach: typing.Tuple[int, EvolutionProblem] = None
        self.approach_params: typing.Dict[str, typing.Any] = None
        self.sensitive_params: typing.Dict[str, typing.Any] = None
        self.workers: typing.List[EvolverWorker] = None
        self.stop_requested: bool = False
        self.last_stop_request = float('-inf')

    def on_interrupt(self, *args) -> None: #pylint: disable=unused-argument
        """Sets stop_requested to True"""
        if not self.stop_requested:
            self.stop_requested = True
            self.logger.critical('SIGINT detected - will stop at the end of the current evolution')
        else:
            stop_from = time.time() - 5000
            if self.last_stop_request > stop_from:
                raise KeyboardInterrupt
            else:
                self.last_stop_request = time.time()
                self.logger.critical('SIGINT suppressed - repeat within 5 seconds to sigterm')



    def run(self) -> None:
        """Main entry to the program. Evolves the network until we run out of time or patience"""
        os.makedirs(self.settings.save_path, exist_ok=True)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)

        handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.settings.save_path, 'main.log'),
            maxBytes=1024*1024, backupCount=5)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p'))
        logger.addHandler(handler)

        self.logger = logger
        del logger

        self.load_or_init()

        old_sigint_handler = signal.signal(signal.SIGINT, self.on_interrupt)
        try:
            self.logger.info('Started')

            self.workers = []
            for i in range(self.settings.cores - 1):
                job_queue = Queue()
                result_queue = Queue()
                proc = Process(target=work,
                               args=(i, self.settings.problem_import,
                                     self.settings.problem_func,
                                     self.settings.save_path,
                                     job_queue,
                                     result_queue))
                proc.daemon = True
                proc.start()
                self.workers.append(
                    EvolverWorker(process=proc, job_queue=job_queue, result_queue=result_queue))

            self.logger.debug('Workers started')
            endtime = time.time() + self.settings.duration_ms / 1000.0

            oldmetric = float('-inf')
            patience_used = 0
            while ((not self.stop_requested)
                   and (time.time() < endtime)
                   and (patience_used < self.settings.evolve_patience)):
                param_ind = np.random.randint(len(self.approach[1]))
                evolve_param = self.approach[1][param_ind]
                self.logger.info('Evolving parameter %s', evolve_param.name)

                context = EvolutionContext(evolve_param)
                context.seed(self)
                while context.choose_check(self):
                    while context.increment_sensitive(self):
                        context.perform_trials(self)
                        while context.select_sensitive_point(self):
                            context.perform_trials(self)
                        context.finish_sensitive(self)
                    context.end_checking(self)
                metric_final, new_value, new_sensitives = context.finish(self)
                self.replace_gen()
                self.generation += 1
                self.sensitive_params = new_sensitives
                self.approach_params[evolve_param.name] = new_value
                self.save()

                if metric_final - self.settings.evolve_epsilon > oldmetric:
                    patience_used = 0
                    oldmetric = metric_final
                else:
                    patience_used += 1

                self.logger.info('Now on generation %s (%s: %s)',
                                 self.generation, self.settings.metric_name, metric_final)


            self.logger.info('Stopping')

            for worker in self.workers:
                worker.job_queue.put(False)

            for worker in self.workers:
                worker.process.join()

            self.logger.info('All workers shutdown')
            logging.shutdown()
        except: #pylint: disable=bare-except
            self.logger.exception('Fatal exception')
            logging.shutdown()
            raise
        finally:
            signal.signal(signal.SIGINT, old_sigint_handler)

    def load_or_init(self) -> None:
        """Loads the saved evolution status if it is available, otherwise initializes each
        parameter randomly.
        """
        if os.path.exists(os.path.join(self.settings.save_path, 'current.json')):
            self.load()
            return

        self.generation = 0
        _approach = self.problem.approaches.sample(None, 1)[0]
        self.approach = (self.problem.approaches.values.index(_approach), _approach)
        del _approach

        self.approach_params = dict()

        self.logger.debug('Initializing generation 0: Approach: %s', str(self.approach[1]))

        longest_name = max(len(i.name) for i in chain(self.approach[1], self.problem.sensitives))

        for cat in self.approach[1].categoricals:
            val = cat.sample(None, 1)[0]
            self.approach_params[cat.name] = val
            self.logger.debug('  %s: %s', cat.name.rjust(longest_name), str(val))
        for i in self.approach[1].integrals:
            val = np.random.randint(i.domain[0], i.domain[1] + 1)
            self.approach_params[i.name] = val
            self.logger.debug('  %s: %s', i.name.rjust(longest_name), str(val))
        for cont in self.approach[1].continuous:
            val = np.random.uniform(cont.domain[0], cont.domain[1])
            self.approach_params[cont.name] = val
            self.logger.debug('  %s: %s', cont.name.rjust(longest_name), str(val))

        self.logger.debug('')

        self.sensitive_params = dict()
        for cat in self.problem.sensitives.categoricals:
            val = cat.sample(None, 1)[0]
            self.sensitive_params[cat.name] = val
            self.logger.debug('  %s: %s', cat.name.rjust(longest_name), str(val))
        for i in self.problem.sensitives.integrals:
            val = np.random.randint(i.domain[0], i.domain[1] + 1)
            self.sensitive_params[i.name] = val
            self.logger.debug('  %s: %s', i.name.rjust(longest_name), str(val))
        for cont in self.problem.sensitives.continuous:
            val = np.random.uniform(cont.domain[0], cont.domain[1])
            self.sensitive_params[cont.name] = val
            self.logger.debug('  %s: %s', cont.name.rjust(longest_name), str(val))

        self.save()

    def load(self):
        """Loads the saved evolution status. Raises OSError if any of the files are missing or
        corrupted. See 'save' for the format used. If the current folder exists, its contents
        are archived to archived/time.time().zip
        """
        current_file = os.path.join(self.settings.save_path, 'current.json')
        with open(current_file, 'r') as infile:
            data = json.load(infile)

        self.generation = data['generation']
        self.approach = (data['approach_ind'], self.problem.approaches[data['approach_ind']])
        self.approach_params = data['approach_params']
        self.sensitive_params = data['sensitive_params']

        current_folder = os.path.join(self.settings.save_path, 'current')
        if os.path.exists(current_folder) and os.listdir(current_folder):
            os.makedirs(os.path.join(self.settings.save_path, 'archived'), exist_ok=True)
            archive_file = os.path.join(self.settings.save_path, 'archived', str(time.time()))
            while os.path.exists(archive_file):
                self.logger.critical('%s already exists.. waiting a second', archive_file)
                time.sleep(1)
                archive_file = os.path.join(self.settings.save_path, 'archived', str(time.time()))

            self.logger.info('Archiving %s to %s', current_folder, archive_file)
            cwd = os.getcwd()
            shutil.make_archive(archive_file, 'zip', current_folder)
            os.chdir(cwd)
            shutil.rmtree(current_folder, onerror=_ignore_del_dir_failure)
            os.chdir(cwd)
            os.makedirs(current_folder, exist_ok=True)

    @staticmethod
    def _clean_params(params: typing.Dict[str, typing.Any]):
        new_params = dict()
        for k, v in params.items():
            if isinstance(v, (int, np.int32, np.int64)):
                new_params[k] = int(v)
            elif isinstance(v, (float, np.float32, np.float64)):
                new_params[k] = float(v)
            else:
                new_params[k] = v
        return new_params

    def save(self):
        """Saves the current settings to the save_path folder. Raises OSError if any of the files
        already exist - we should archive all these settings at each generation (see 'replace_gen'
        for details). We would do the same for folders, but that's painful on windows since it means
        having the explorer open can cause the program to fail

        Settings:
            save_path/
                current.json: a dict containing generation, approach_ind, approach_params,
                              and sensitive_params
                current/
                    A collection of plots about the current generation (these will be saved
                    throughout the generation and the folder is the only thing initialized
                    here)
        """
        os.makedirs(self.settings.save_path, exist_ok=True)
        current_file = os.path.join(self.settings.save_path, 'current.json')
        if os.path.exists(current_file):
            raise FileExistsError()
        current_folder = os.path.join(self.settings.save_path, 'current')
        os.makedirs(current_folder, exist_ok=True)

        tosave = {
            'generation': self.generation,
            'approach_ind': self.approach[0],
            'approach_params': Evolver._clean_params(self.approach_params),
            'sensitive_params': Evolver._clean_params(self.sensitive_params)
        }

        with open(current_file, 'w') as outfile:
            json.dump(tosave, outfile)


    def replace_gen(self):
        """Replaces the current generation. Takes the current.json and
        current/ folder and archives it into history/gen{generation}/, deletes current.json,
        and deletes the current/ folder as far as possible, ignoring failures to delete directories
        due to windows not handling that well
        """
        current_path = os.path.join(self.settings.save_path, 'current.json')
        current_folder_path = os.path.join(self.settings.save_path, 'current')
        history_path = os.path.join(self.settings.save_path, 'history')
        archive_folder_path = os.path.join(history_path, f'gen{self.generation}')
        archive_path = os.path.join(archive_folder_path, 'current') # no ending allowed
        archive_json_path = os.path.join(archive_folder_path, 'current.json')


        if not os.path.exists(current_path):
            raise FileNotFoundError
        if not os.path.exists(current_folder_path):
            raise FileNotFoundError

        os.makedirs(history_path, exist_ok=True)
        os.makedirs(archive_folder_path)

        cwd = os.getcwd()
        shutil.make_archive(archive_path, 'zip', current_folder_path)
        os.chdir(cwd)
        shutil.rmtree(current_folder_path, onerror=_ignore_del_dir_failure)
        os.chdir(cwd)

        os.rename(current_path, archive_json_path)

