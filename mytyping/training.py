"""Training injectables"""

import torch
from shared.ssptrainer import SSPGenericTrainingContext
import mytyping.encoding as menc

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

    def measure(self, context: SSPGenericTrainingContext) -> None:
        """Measures accuracy and updates last_measure_epoch and accuracy"""
        ssp = context.test_ssp if self.validation else context.train_ssp

        real_num_points = min(self.num_points, ssp.epoch_size)
        correct_preds = 0
        ssp.mark()
        for _ in range(real_num_points):
            inp, out = next(ssp)
            act = context.teacher.classify_many(context.model, [inp])[0]
            correct_preds += menc.accuracy(act, out)

        ssp.reset()

        accuracy = float(correct_preds) / float(real_num_points)

        self.last_measure_epoch = context.shared['epochs'].epochs
        self.accuracy = accuracy
        context.logger.info('[AccuracyTracker] %s/%s (%s%%)', int(correct_preds), int(real_num_points), f'{float(accuracy*100):.2f}')

    def setup(self, context: SSPGenericTrainingContext, **kwargs) -> None: #pylint: disable=unused-argument
        """Stores self into context.shared['accuracy']"""
        context.shared['accuracy'] = self

    def pre_loop(self, context: SSPGenericTrainingContext) -> None:
        """Measures accuracy if necessary"""
        epoch_tracker = context.shared['epochs']
        if (epoch_tracker.new_epoch
                and int(epoch_tracker.epochs) - self.measure_every >= self.last_measure_epoch):
            self.measure(context)

    def finished(self, context: SSPGenericTrainingContext, result: dict) -> None:
        """Remeasures and stores accuracy"""
        self.measure(context)
        result['accuracy'] = self.accuracy