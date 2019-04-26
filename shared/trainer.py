"""This module handles the generic trainer, which is a very extensible method of training
all types of networks"""

import typing
import torch
import torch.nn
import torch.optim
import logging
import math
import os
import shutil
import time

from shared.models.generic import Network
from shared.events import Event
from shared.pwl import PointWithLabelProducer
from shared.teachers import NetworkTeacher
from shared.filetools import zipdir
from shared.weight_inits import WeightInitializer

class GenericTrainingContext(typing.NamedTuple):
    """Describes the training context. Acts a store for the common variables we need in
    read-only manner. The context may be replaced within-training

    Attributes:
        model (Network): the network which is being trained
        teacher (NetworkTeacher): the teacher for the network
        train_pwl (PointWithLabelProducer): the producer that is generating the training points
        test_pwl (PointWithLabelProducer): the producer that should generate the validation points
        optimizer (torch.optim.Optimizer): the optimizer

        batch_size (int): the batch size we are currently using

        points (tensor, batch_size x model.input_dim)
        labels (tensor, batch_size x model.output_dim)

        shared (dict): This may be used to exchange transient information
        logger (logging.Logger): the logger that should be used
    """

    model: Network
    teacher: NetworkTeacher
    train_pwl: PointWithLabelProducer
    test_pwl: PointWithLabelProducer
    optimizer: torch.optim.Optimizer

    batch_size: int
    points: torch.tensor
    labels: torch.tensor
    shared: dict
    logger: logging.Logger

class GenericTrainer:
    """Defines a generic trainer which uses events to add functionality.

    Attributes:
        train_pwl (PointWithLabelProducer): the producer of training points
        test_pwl (PointWithLabelProducer): the producer of validation points
        teacher (NetworkTeacher): the instance actually capable of sending points
            to the network

        batch_size (int): the initial batch size when training
        learning_rate (double): the initial learning rate for all parameters
        optimizer (torch.nn.optim.Optimizer): the optimizer to use
        criterion (torch.nn.modules.loss._Loss): typically torch.nn.CrossEntropyLoss

        setup (Event(*(ctx,), **kwargs)): called at the beginning of training, right after
            constructing the context. Passed keyword arguments to trainer.train as well as the
            generic training context
        pre_loop (Event(ctx)): invoked at the beginning of each loop, before fetching points.
        post_points (Event(ctx)): invoked after fetching the points we are going to feed the network
        pre_train (Event(ctx)): invoked immediately prior to training
        post_train (Event(ctx, loss)): invoked immedaitely after training with the loss
        decay_scheduler (Event(ctx, loss, result) -> bool): invoked after post_train. it is passed
            the loss this loop. The result starts at False and for future decay_schedulers is the
            result of the previous scheduler.
        decay (Event(ctx) -> GenericTrainerContext): invoked if decay_scheduler returns true.
            returns to new training context, or the old training context if the decayer did not
            change it
        stopper (Event(ctx) -> bool): invoked at the end of each loop. if any subscribers return
            true, training ends.
        finished (Event(ctx, dict)): invoked after everything is finished. The dict passed is
            returned from train()
    """

    def __init__(self, train_pwl: PointWithLabelProducer, test_pwl: PointWithLabelProducer,
                 teacher: NetworkTeacher, batch_size: int, learning_rate: float,
                 optimizer: torch.optim.Optimizer, criterion: 'Loss'):
        self.train_pwl = train_pwl
        self.test_pwl = test_pwl
        self.teacher = teacher

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = criterion

        self.setup = Event('setup')
        self.pre_loop = Event('pre_loop')
        self.post_points = Event('post_points')
        self.pre_train = Event('pre_train')
        self.post_train = Event('post_train')
        self.decay_scheduler = Event('decay_scheduler', lambda args, kwargs, result: (True, (args[0], args[1], result), kwargs))
        self.decay = Event('decay', lambda args, kwargs, result: (True, (result,), kwargs))
        self.stopper = Event('stopper', lambda args, kwargs, result: (not result, args, kwargs))
        self.finished = Event('finished')

    def register_classlike(self, obj: typing.Any) -> 'GenericTrainer':
        """Treats the given object as a class. If it has any members that match an event we have,
        that member is added as a subscriber to that event. If it has a 'delegated_object' member,
        that member is used to fill any events that the obj does not itself have."""

        delegates = [obj]
        cur = obj
        while hasattr(cur, 'delegated_object'):
            cur = cur.delegated_object
            delegates.append(cur)
        del cur

        for key in {'setup', 'pre_loop', 'post_points', 'pre_train', 'post_train',
                    'decay_scheduler', 'decay', 'stopper', 'finished'}:
            for deleg in delegates:
                if hasattr(deleg, key):
                    getattr(self, key).__iadd__(getattr(deleg, key))
                    break
        return self

    def reg(self, obj: typing.Any) -> 'GenericTrainer':
        """Alias for register_classlike"""
        return self.register_classlike(obj)

    def train(self, model: Network, **kwargs):
        """Trains the given network. Returns an output dict. The kwargs are passed to the setup
        functions. If 'logger' is in kwargs, it is used as the logger and not passed further"""

        if 'logger' in kwargs:
            _logger = kwargs['logger']
            del kwargs['logger']
        else:
            _logger = logging.getLogger(__name__)
            _logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

        context = GenericTrainingContext(
            model=model, teacher=self.teacher, train_pwl=self.train_pwl, test_pwl=self.test_pwl,
            batch_size=self.batch_size, optimizer=self.optimizer,
            points=torch.zeros((self.batch_size, model.input_dim), dtype=torch.double),
            labels=torch.zeros(self.batch_size, dtype=torch.long), shared=dict(),
            logger=_logger
        )
        del _logger

        for param_group in context.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        self.setup(context, **kwargs)

        while True:
            self.pre_loop(context)
            context.train_pwl.fill(context.points, context.labels)
            self.post_points(context)
            self.pre_train(context)
            loss = context.teacher.teach_many(context.model, self.optimizer, self.criterion,
                                              context.points, context.labels)
            self.post_train(context, loss)
            if self.decay_scheduler(context, loss, False):
                context = self.decay(context)
            if self.stopper(context):
                break

        result = dict()
        self.finished(context, result)
        return result

class EpochsTracker:
    """Tracks the number of epochs we've gone through. Sets context.shared['epochs'] to this
    and saves epochs in the output.

    Attributes:
        epochs (float): the number of epochs we've gone through (correct from post_train (inclusive) to pre_loop (exclusive))
        new_epoch (bool): True if this is a new epoch, false otherwise
        first (bool): True if this is the first loop, false otherwise
    """

    def __init__(self):
        self.epochs = 0.0
        self.new_epoch = False
        self.first = True

    def setup(self, context, **kwargs):
        """Initializes epochs to 0 and new_epoch to false"""
        context.shared['epochs'] = self
        context.logger.debug('[EpochsTracker] starting epoch 0')
        self.epochs = 0.0
        self.new_epoch = False
        self.first = True

    def pre_loop(self, context: GenericTrainingContext) -> None:
        """Updates epochs and new_epoch"""
        self.first = self.epochs == 0
        future_pos = context.train_pwl.position + context.batch_size
        if future_pos >= context.train_pwl.epoch_size:
            future_pos -= context.train_pwl.epoch_size
            new_epoch = math.floor(self.epochs) + 1
            self.epochs = new_epoch + (future_pos / context.train_pwl.epoch_size)
            self.new_epoch = True
            context.logger.debug('[EpochsTracker] starting epoch %s', new_epoch)
        else:
            self.epochs = math.floor(self.epochs) + (future_pos / context.train_pwl.epoch_size)
            self.new_epoch = False

    def finished(self, context: GenericTrainingContext, result: dict) -> None: # pylint: disable=unused-argument
        """Saves the number of epochs to the result"""
        result['epochs'] = self.epochs

class EpochsStopper:
    """Stops after a certain number of epochs. Requires an EpochsTracker

    Attributes:
        stop_after (int): the number of epochs to stop after
    """

    def __init__(self, stop_after: int):
        self.stop_after = stop_after

    def stopper(self, context: GenericTrainingContext) -> bool:
        """Returns true if self.stop_after epochs have passed"""
        return context.shared['epochs'].epochs >= self.stop_after

class OnEpochCaller:
    """Calls a particular function after a certain number of epochs have passed
    with the current training context. Requires an EpochsTracker and expects to
    be after it.

    Attributes:
        epoch_filter (callable -> bool): returns the filter for the epoch.
            Accepts the numeric epoch and returns True if on_epoch should be
            called and False otherwise. several defaults are available through
            the create_ methods.
        on_epoch (callable): the object to call with the context and a filename
            hint
    """

    def __init__(self, epoch_filter: typing.Callable, on_epoch: typing.Callable):
        if not callable(epoch_filter):
            raise ValueError(f'expected epoch_filter is callable, got {epoch_filter} (type={type(epoch_filter)})')
        if not callable(on_epoch):
            raise ValueError(f'expected on_epoch is callable, got {on_epoch} (type={type(on_epoch)})')

        self.epoch_filter = epoch_filter
        self.on_epoch = on_epoch

    @classmethod
    def create_every(cls, on_epoch: typing.Callable,
                     start=0, skip=1, stop=None) -> 'OnEpochCaller':
        """Calls the given callable every skip epochs after start, not including
        the stop epoch or any epoch after that.

        Args:
            on_epoch (typing.Callable): the thing to do
            start (int, optional): Defaults to 0. the first epoch which on_epoch is called
            skip (int, optional): Defaults to 1. the number of epochs to wait between on_epoch
            stop (int, optional): Defaults to None. if set, on_epoch is not called at stop or
                later epochs

        Returns:
            caller (OnEpochCaller): the classlike object to register with the trainer
        """

        def epoch_filter(epoch: int):
            if epoch < start:
                return False
            if (epoch - start) % skip != 0:
                return False
            if stop is not None and epoch >= stop:
                return False
            return True
        return cls(epoch_filter, on_epoch)

    def pre_loop(self, context: GenericTrainingContext):
        """Checks epoch"""
        tracker = context.shared['epochs']
        if (tracker.first or tracker.new_epoch) and self.epoch_filter(int(tracker.epochs)):
            self.on_epoch(context, f'epoch_{int(tracker.epochs)}')

    def finished(self, context: GenericTrainingContext, result: dict) -> None: # pylint: disable=unused-argument
        """Saves final_epoch"""
        self.on_epoch(context, 'epoch_finished')

class OnFinishCaller:
    """Calls a particular function when finished"""
    def __init__(self, on_finished):
        self.finished = on_finished

class DecayOnPlateau:
    """Decays the loss if there has been no improvement in a certain number of epochs.
    Requires an EpochsTracker

    Attributes:
        patience (int): the number of epochs we wait for improvement
        improve_epsilon (float): the minimum improvement we consider

        patience_used (int): the number of epochs we've gone through without improvement
        best_loss (float): the best loss we have seen
        improved_loss (bool): if we improved the loss this epoch

        loop_loss (float): the best loss we've seen this loop
    """

    def __init__(self, patience: int = 5, improve_epsilon: float = 1e-06):
        self.patience = patience
        self.improve_epsilon = improve_epsilon

        self.patience_used = 0
        self.best_loss = float('inf')
        self.improved_loss = False

        self.loop_loss = float('inf')

    def decay_scheduler(self, context: GenericTrainingContext, loss: float, result: bool) -> bool:
        """Decays if the loss hasn't improved in self.patience epochs"""

        epochs_tracker: EpochsTracker = context.shared['epochs']
        if epochs_tracker.new_epoch:
            if not self.improved_loss:
                self.patience_used += 1
                context.logger.debug('[DecayOnPlateau] used up patience #%s'
                                     + ' (best loss this loop: %s)',
                                     self.patience_used, self.loop_loss)
                self.loop_loss = float('inf')
                if self.patience_used >= self.patience:
                    self.patience_used = 0
                    return True
            else:
                self.loop_loss = float('inf')
                self.patience_used = 0
                self.improved_loss = False
        else:
            if loss < self.best_loss - self.improve_epsilon:
                context.logger.info('[DecayOnPlateau] improved loss to %s', loss)
                self.improved_loss = True
                self.best_loss = loss
            if loss < self.loop_loss:
                self.loop_loss = loss

        return result

class DecayTracker:
    """Keeps track of the number of decays that occurred. Sets context.shared['decays'] to this.

    Attributes:
        decays (int): the number of decays that have occurred
    """

    def __init__(self):
        self.decays = 0

    def setup(self, context, **kwargs):
        """Sets context.shared['decays'] to self"""
        context.shared['decays'] = self

    def decay(self, context: GenericTrainingContext) -> GenericTrainingContext:
        """Increments decays counter"""
        self.decays += 1
        context.logger.info('[DecaysTracker] this is decay %s', self.decays)
        return context

    def finished(self, context: GenericTrainingContext, result: dict) -> None: #pylint: disable=unused-argument
        """Saves the number of decays"""
        result['decays'] = self.decays

class LRMultiplicativeDecayer:
    """Decays the learning rate by multiplying it by a constant

    Attributes:
        factor (float): the multipicative factor for the learning rate
        minimum (float): the minimum learning rate
    """

    def __init__(self, factor=0.5, minimum=1e-09):
        self.factor = factor
        self.minimum = minimum

    def decay(self, context: GenericTrainingContext) -> GenericTrainingContext:
        """Performs multiplicative LR decay"""
        lowest_lr = 1000
        for param_group in context.optimizer.param_groups:
            param_group['lr'] *= self.factor
            if param_group['lr'] < self.minimum:
                param_group['lr'] = self.minimum
            lowest_lr = min(param_group['lr'], lowest_lr)
        context.logger.debug('[LRMultiplicativeDecayer] learning rate=%s', lowest_lr)
        return context

class DecayStopper:
    """Stops after a certain number of decays have occurred. Requires a DecayTracker

    Attributes:
        stop_after (int): the number of decays to stop after
    """

    def __init__(self, stop_after: int):
        self.stop_after = stop_after

    def stopper(self, context: GenericTrainingContext) -> bool:
        """Stops after stop_after decays"""
        return context.shared['decays'].decays >= self.stop_after

class AccuracyTracker:
    """Measures accuracy every x epochs. Puts self into context.shared['accuracy'].

    Attributes:
        measure_every (int): the number of epochs between accuracy measures
        num_points (int): the number of points to measure
        validation (bool): true for a validation measure, false for a train data measure

        last_measure_epoch (int): the last epoch we measured at
        accuracy (float): the accuracy percentage on the last check
    """
    def __init__(self, measure_every: int, num_points: int, validation: bool):
        self.measure_every = measure_every
        self.num_points = num_points
        self.validation = validation

        self.last_measure_epoch = float('-inf')
        self.accuracy = float('nan')

    def measure(self, context: GenericTrainingContext) -> None:
        """Measures accuracy and updates last_measure_epoch and accuracy"""
        pwl = context.test_pwl if self.validation else context.train_pwl

        real_num_points = int(math.ceil(self.num_points / context.batch_size) * context.batch_size)

        result = torch.zeros((context.batch_size, context.model.output_dim), dtype=torch.double)
        correct_preds = 0
        pwl.mark()
        for _ in range(real_num_points // context.batch_size):
            pwl.fill(context.points, context.labels)

            context.teacher.classify_many(context.model, context.points, result)
            real_guesses = result.argmax(1)
            correct_preds += (real_guesses == context.labels).sum()
        pwl.reset()

        accuracy = float(correct_preds) / float(real_num_points)

        self.last_measure_epoch = context.shared['epochs'].epochs
        self.accuracy = accuracy
        context.logger.info('[AccuracyTracker] %s/%s (%s%%)', correct_preds, real_num_points, accuracy)

    def setup(self, context: GenericTrainingContext, **kwargs) -> None: #pylint: disable=unused-argument
        """Stores self into context.shared['accuracy']"""
        context.shared['accuracy'] = self

    def pre_loop(self, context: GenericTrainingContext) -> None:
        """Measures accuracy if necessary"""
        epoch_tracker = context.shared['epochs']
        if (epoch_tracker.new_epoch
                and int(epoch_tracker.epochs) - self.measure_every >= self.last_measure_epoch):
            self.measure(context)

    def finished(self, context: GenericTrainingContext, result: dict) -> None:
        """Remeasures and stores accuracy"""
        self.measure(context)
        result['accuracy'] = self.accuracy
class WeightNoiser:
    """Adds some noise to the last->output weights during training

    Attributes:
        noise (WeightInitializer): the noise initialization
        tensor_fetcher (callable): a function that accepts the GenericTrainingContext
            and returns the tensor to noise
        current_noise (torch.tensor [like tensor_fetcher result]):
            the underlying noise tensor we store noise in during training
    """

    def __init__(self, noise: WeightInitializer, tensor_fetcher: typing.Callable):
        self.noise = noise
        self.tensor_fetcher = tensor_fetcher
        self.current_noise = None

    def setup(self, context: GenericTrainingContext, **kwargs) -> None:
        """Initializes the noise tensor like the tensor_fetcher"""
        to_noise = self.tensor_fetcher(context)
        self.current_noise = torch.zeros_like(to_noise)

    def pre_train(self, context: GenericTrainingContext):
        """Updates the current noise and applies it"""
        self.current_noise[:] = 0
        self.noise.initialize(self.current_noise)
        self.tensor_fetcher(context)[:] += self.current_noise

class ZipDirOnFinish:
    """Zips a particular directory to directory + .zip when finished. Deletes the
    zip if it already exists

    Attributes:
        dirpath (str): the directory to zip
    """

    def __init__(self, dirpath: str):
        if not isinstance(dirpath, str):
            raise ValueError(f'expected dirpath is str, got {dirpath} (type={type(dirpath)})')
        self.dirpath = dirpath

    def finished(self, context: GenericTrainingContext, result: dict) -> None: #pylint: disable=unused-argument
        """Zips the directory"""
        if not os.path.exists(self.dirpath):
            return

        if os.path.exists(self.dirpath + '.zip'):
            os.remove(self.dirpath + '.zip')
        zipdir(self.dirpath)

class CopyLogOnFinish:
    """Copies log.txt to the output directory upon finishing

    Attributes:
        outpath (str): the path to where the logfile should be saved
    """

    def __init__(self, outpath: str):
        if not isinstance(outpath, str):
            raise ValueError(f'expected outpath is str, got {outpath} (type={type(outpath)})')
        self.outpath = outpath

    def finished(self, context: GenericTrainingContext, result: dict): # pylint: disable=unused-argument
        """Copies the log file"""
        if not os.path.exists('log.txt'):
            context.logger.info('[CopyLogOnFinish] Skipping because log.txt does not exist')
            return

        shutil.copy('log.txt', self.outpath)

class EpochProgress:
    """A more complicated version of epoch caller that will print out every 15 seconds
    how far into the epoch we are. Helpful when epochs take a long time

    Attributes:
        print_every (float): number of seconds between prints
        last_print (float): when we last printed progress
        last_epoch (float): the value of epochs the last time we printed
    """

    def __init__(self, print_every=15):
        self.print_every = float(print_every)
        self.last_print = None
        self.last_epoch = None

    def print(self, context: GenericTrainingContext):
        """Prints out progress information"""

        epoch = context.shared['epochs'].epochs
        thetime = time.time()

        progress = epoch - self.last_epoch
        duration = thetime - self.last_print

        seconds_per_epoch = duration / progress
        time_left_in_epoch = (epoch - int(epoch)) * seconds_per_epoch

        context.logger.info(f'[EpochProgress] Epoch {epoch:.2f} ({progress:.2f} in last {duration:.2f}s, {seconds_per_epoch:.2f} secs/epoch, {time_left_in_epoch:.2f} secs rem in epoch)')

    def pre_train(self, context: GenericTrainingContext):
        """Determines if we should print out progress"""
        printed = False
        if self.last_print is None:
            printed = True
        elif time.time() - self.last_print > self.print_every:
            self.print(context)
            printed = True

        if printed:
            self.last_print = time.time()
            self.last_epoch = context.shared['epochs'].epochs


def save_model(outpath: str):
    """A callable for EpochCaller and similar that saves the model to the given
    folder"""

    os.makedirs(outpath)

    def on_step(context: GenericTrainingContext, fname_hint: str):
        outfile = os.path.join(outpath, fname_hint + '.pt')
        context.logger.info('[SAVE] Saving model to %s', outfile)
        torch.save(context.model, outfile)

    return on_step
