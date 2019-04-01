"""Handles setting up an rnn for evolution. Will give you everything except the problem to
solve"""

import typing
import shared.evolution as evo
from shared.models.rnn import NaturalRNN, RNNTeacher
import shared.weight_inits as wi
import shared.trainer as tnr
import torch

class RNNApproach(evo.EvolutionProblem):
    """Describes an RNN approach to solving a classification problem.

    Attributes:
        pwl_func (callable): returns the (train, validation) point with label producers
    """

    def __init__(self, pwl_func: typing.Callable):
        super().__init__(
            categoricals=[
                evo.CategoricalEvolvableParam('nonlinearity', ['tanh', 'relu'])
            ],
            integrals=[
                evo.NumericEvolvableParam('hidden_size', True, (50, 250)),
                evo.NumericEvolvableParam('recurrent_times', True, (1, 50))
            ],
            continuous=[
                evo.NumericEvolvableParam('g', False, (1.0, 1000.0)),
                evo.NumericEvolvableParam('dt', False, (1e-06, 0.01)),
                evo.NumericEvolvableParam('inp_stddev', False, (1e-06, 0.1)),
                evo.NumericEvolvableParam('hidden_bias_vari', False, (0.1, 1.0)),
                evo.NumericEvolvableParam('output_weight_vari', False, (0.1, 1.0)),
                evo.NumericEvolvableParam('alpha', False, (0.8, 0.99)),
                evo.NumericEvolvableParam('lr_factor', False, (0.1, 0.8))
            ]
        )

        self.pwl_func = pwl_func

    def realize(self, values: typing.Dict[str, typing.Any], **sensitives):
        train_pwl, test_pwl = self.pwl_func()

        network = NaturalRNN.create(
            str(values['nonlinearity']), test_pwl.input_dim, int(values['hidden_size']), test_pwl.output_dim,
            input_weights=wi.OrthogonalWeightInitializer(float(values['inp_stddev']), 0),
            input_biases=wi.ZerosWeightInitializer(),
            hidden_weights=wi.SompolinskySmoothedFixedGainWeightInitializer(
                float(values['dt']), float(values['g'])),
            hidden_biases=wi.GaussianWeightInitializer(
                mean=0, vari=float(values['hidden_bias_vari']), normalize_dim=0),
            output_weights=wi.GaussianWeightInitializer(
                mean=0, vari=float(values['output_weight_vari']), normalize_dim=0),
            output_biases=wi.ZerosWeightInitializer()
        )

        trainer = tnr.GenericTrainer(
            train_pwl=train_pwl,
            test_pwl=test_pwl,
            teacher=RNNTeacher(recurrent_times=int(values['recurrent_times']), input_times=1),
            batch_size=int(sensitives['batch_size']),
            learning_rate=float(sensitives['learning_rate']),
            optimizer=torch.optim.RMSprop(
                [p for p in network.parameters() if p.requires_grad],
                lr=0.001, alpha=float(values['alpha'])),
            criterion=torch.nn.CrossEntropyLoss()
        )

        (trainer
         .reg(tnr.EpochsTracker())
         .reg(tnr.EpochsStopper(150))
         .reg(tnr.DecayTracker())
         .reg(tnr.DecayStopper(8))
         .reg(tnr.LRMultiplicativeDecayer(factor=values['lr_factor']))
         .reg(tnr.DecayOnPlateau())
         .reg(tnr.AccuracyTracker(5, 1000, True))
        )

        return trainer, network
