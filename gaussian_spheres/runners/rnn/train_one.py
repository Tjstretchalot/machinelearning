"""This runner trains a single recurrent network on the gaussian spheres task"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.rnn import NaturalRNN, RNNTeacher
import shared.trainer as tnr
import shared.weight_inits as wi
import shared.measures.pca as pca
import shared.measures.dist_through_time as dtt
import torch
from gaussian_spheres.pwl import GaussianSpheresPWLP
import os

SAVEDIR = os.path.join('out', 'gaussian_spheres', 'runners', 'rnn', 'train_one')

def main():
    """Entry point"""
    pwl = GaussianSpheresPWLP.create(
        epoch_size=1800, input_dim=200, output_dim=2, cube_half_side_len=2,
        num_clusters=60, std_dev=0.04, mean=0, min_sep=0.1
    )

    network = NaturalRNN.create(
        'tanh', pwl.input_dim, 200, pwl.output_dim,
        input_weights=wi.OrthogonalWeightInitializer(0.03, 0),
        input_biases=wi.ZerosWeightInitializer(), #
        hidden_weights=wi.SompolinskySmoothedFixedGainWeightInitializer(0.001, 20),
        hidden_biases=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=0),
        output_weights=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=0),
        output_biases=wi.ZerosWeightInitializer()
    )

    trainer = tnr.GenericTrainer(
        train_pwl=pwl,
        test_pwl=pwl,
        teacher=RNNTeacher(recurrent_times=10, input_times=1),
        batch_size=30,
        learning_rate=0.05,
        optimizer=torch.optim.RMSprop([p for p in network.parameters() if p.requires_grad], lr=0.001, alpha=0.9),
        criterion=torch.nn.CrossEntropyLoss()
    )

    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(150))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
    )

    print('--saving pcs before training--')
    traj = pca.find_trajectory(network, pwl, 10, 2)

    print('--saving distance through time before training--')
    savepath = os.path.join(SAVEDIR, 'dtt_before')
    dtt.measure_dtt(network, pwl, 10, savepath, verbose=True, exist_ok=True)

    savepath = os.path.join(SAVEDIR, 'pca_before')
    pca.plot_trajectory(traj, savepath, exist_ok=True)
    del traj

    print('--training--')
    result = trainer.train(network)
    print('--finished training--')
    print(result)
    print('--saving pcs after training--')
    traj = pca.find_trajectory(network, pwl, 10, 2)

    print('--saving distance through time after training--')
    savepath = os.path.join(SAVEDIR, 'dtt_after')
    dtt.measure_dtt(network, pwl, 10, savepath, verbose=True, exist_ok=True)

    savepath = os.path.join(SAVEDIR, 'pca_after')
    pca.plot_trajectory(traj, savepath, exist_ok=True)

if __name__ == '__main__':
    main()
