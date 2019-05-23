"""This module is like the trainer except for sequence to sequence training. It is as compatible as
possible"""

import typing
import torch
import logging
from shared.models.generic import Network
from shared.seqseqprod import SeqSeqProducer
from shared.teachers import SeqSeqTeacher
from shared.events import Event
from shared.perf_stats import PerfStats, NoopPerfStats, LoggingPerfStats
import shared.trainer

class SSPGenericTrainingContext(typing.NamedTuple):
    """The sequence-sequence generic training context. Analogous to a standard
    training context, except we don't reuse tensors for these sequences

    Attributes:
        model (Network): the network that we are training
        teacher (SeqSeqTeacher): the teacher we are using

        train_ssp (SeqSeqProducer): the training data
        test_ssp (SeqSeqProducer): the test data
        optimizers (list[torch.optim.Optimizer]): the optimizers

        batch_size (int): the batch size we are currently using

        shared (dict): this may be used to exchange information
        logger (logging.Logger): the logger
    """

    model: Network
    teacher: SeqSeqTeacher

    train_ssp: SeqSeqProducer
    test_ssp: SeqSeqProducer
    optimizers: typing.List[torch.optim.Optimizer]

    batch_size: int

    shared: dict
    logger: logging.Logger
    perf_stats: PerfStats

    @property
    def train_pwl(self):
        return self.train_ssp

    @property
    def test_pwl(self):
        return self.test_ssp

class SSPGenericTrainer:
    """A generic trainer for sequence to sequence models

    Attributes:
        train_ssp (SeqSeqProducer): the training data
        test_ssp (SeqSeqProducer): the validation data
        teacher (SeqSeqTeacher): the network teacher
        batch_size (int): the starting batch size
        learning_rate (float, optional): the default learning rate or None to not change.

        optimizers (list[torch.optim.Optimizer]): the optimizers that this trainer uses
        criterion (any): the loss that is used to evaluate the networks response

        setup (Event(*(ctx,), **kwargs)): called at the beginning of training, right after
            constructing the context. Passed keyword arguments to trainer.train as well as the
            generic training context
        pre_loop (Event(ctx)): invoked at the beginning of each loop, before fetching points.
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
    def __init__(self, train_ssp: SeqSeqProducer, test_ssp: SeqSeqProducer,
                 teacher: SeqSeqTeacher, batch_size: int, learning_rate: float,
                 optimizers: typing.List[torch.optim.Optimizer], criterion: 'Loss'):
        self.train_ssp = train_ssp
        self.test_ssp = test_ssp
        self.teacher = teacher

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizers = optimizers
        self.criterion = criterion

        self.setup = Event('setup')
        self.pre_loop = Event('pre_loop')
        self.pre_train = Event('pre_train')
        self.post_train = Event('post_train')
        self.decay_scheduler = Event('decay_scheduler', lambda args, kwargs, result: (True, (args[0], args[1], result), kwargs))
        self.decay = Event('decay', lambda args, kwargs, result: (True, (result,), kwargs))
        self.stopper = Event('stopper', lambda args, kwargs, result: (not result, args, kwargs))
        self.finished = Event('finished')

    def register_classlike(self, obj: typing.Any) -> 'SSPGenericTrainer':
        """Treats the given object as a class. If it has any members that match an event we have,
        that member is added as a subscriber to that event. If it has a 'delegated_object' member,
        that member is used to fill any events that the obj does not itself have."""

        delegates = [obj]
        cur = obj
        while hasattr(cur, 'delegated_object'):
            cur = cur.delegated_object
            delegates.append(cur)
        del cur

        for key in {'setup', 'pre_loop', 'pre_train', 'post_train',
                    'decay_scheduler', 'decay', 'stopper', 'finished'}:
            for deleg in delegates:
                if hasattr(deleg, key):
                    getattr(self, key).__iadd__(getattr(deleg, key))
                    break
        return self

    def reg(self, obj: typing.Any) -> 'SSPGenericTrainer':
        """Alias for register_classlike"""
        return self.register_classlike(obj)

    def train(self, model: Network, **kwargs):
        """Trains the given sequence to sequence model"""

        if 'logger' in kwargs:
            _logger = kwargs['logger']
            del kwargs['logger']
        else:
            _logger = logging.getLogger(__name__)
            _logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

        if 'perf_stats' in kwargs:
            perf_stats = kwargs['perf_stats']
            del kwargs['perf_stats']
        else:
            perf_stats = NoopPerfStats()

        context = SSPGenericTrainingContext(
            model=model, teacher=self.teacher, train_ssp=self.train_ssp, test_ssp=self.test_ssp,
            batch_size=self.batch_size, optimizers=self.optimizers, shared=dict(),
            logger=_logger, perf_stats=perf_stats
        )
        del _logger
        del perf_stats

        if self.learning_rate is not None:
            for optim in context.optimizers:
                for param_group in optim.param_groups:
                    param_group['lr'] = self.learning_rate

        self.setup(context, **kwargs)

        while True:
            context.perf_stats.enter('PRE_LOOP')
            self.pre_loop(context)
            context.perf_stats.exit_then_enter('GET_INPUTS')

            inputs = []
            outputs = []
            for _ in range(context.batch_size):
                inp, out = next(context.train_ssp)
                inputs.append(inp)
                outputs.append(out)

            context.perf_stats.exit_then_enter('PRE_TRAIN')
            self.pre_train(context)
            context.perf_stats.exit_then_enter('TEACH_MANY')
            loss = context.teacher.teach_many(context.model, context.optimizers, self.criterion,
                                              inputs, outputs, context.perf_stats)
            context.perf_stats.exit_then_enter('POST_TRAIN')
            self.post_train(context, loss)
            context.perf_stats.exit_then_enter('DECAY_SCHEDULER')
            if self.decay_scheduler(context, loss, False):
                context.perf_stats.exit_then_enter('DECAY')
                context = self.decay(context)
            context.perf_stats.exit_then_enter('STOPPER')
            if self.stopper(context):
                context.perf_stats.exit()
                break
            context.perf_stats.exit()

        result = dict()
        context.perf_stats.enter('FINISHED')
        self.finished(context, result)
        context.perf_stats.exit()
        return result
