"""This module handles the generic trainer, which is a very extensible method of training
all types of networks"""

import typing
import torch
import torch.nn
import torch.optim
import logging
import math

from shared.models.generic import Network
from shared.events import Event
from shared.pwl import PointWithLabelProducer
from shared.teachers import NetworkTeacher

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
    """

    def __init__(self):
        self.epochs = 0.0
        self.new_epoch = False

    def setup(self, context, **kwargs):
        """Initializes epochs to 0 and new_epoch to false"""
        context.shared['epochs'] = self
        context.logger.debug('[EpochsTracker] starting epoch 0')
        self.epochs = 0.0
        self.new_epoch = False

    def pre_loop(self, context: GenericTrainingContext) -> None:
        """Updates epochs and new_epoch"""
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
