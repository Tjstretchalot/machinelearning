"""Trains a single feedforward model on the mnist task
"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.ff import FeedforwardLarge, FFTeacher
import shared.trainer as tnr
import shared.weight_inits as wi
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import torch
from mnist.pwl import MNISTData
import os

FOLDER_NAME = os.path.splitext(os.path.basename(__file__))[0]
SAVEDIR = os.path.join('out', 'mnist', 'runners', FOLDER_NAME)

def main():
    """Entry point"""
    train_pwl = MNISTData.load_train().to_pwl().restrict_to(set(range(10))).rescale()
    test_pwl = MNISTData.load_test().to_pwl().restrict_to(set(range(10))).rescale()

    network = FeedforwardLarge.create(
        input_dim=train_pwl.input_dim, output_dim=train_pwl.output_dim, nonlinearity='none',
        weights=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=1),
        biases=wi.ZerosWeightInitializer(),
        layer_sizes=[200, 100, 50, 50, 50, 25]
        #layer_sizes=[500, 200]
    )

    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=FFTeacher(),
        batch_size=30,
        learning_rate=0.001,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.003),
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

    print('--saving distance through layers before training--')
    savepath = os.path.join(SAVEDIR, 'dtt_before_train')
    dtt.measure_dtt_ff(network, train_pwl, savepath, verbose=True, exist_ok=True)

    print('--saving pca before training--')
    savepath = os.path.join(SAVEDIR, 'pca_before_train')
    traj = pca_ff.find_trajectory(network, train_pwl, 2)
    pca_ff.plot_trajectory(traj, savepath, exist_ok=True, alpha=1.0)
    del traj

    trainer.train(network)

    print('--saving distance through layers after training (train)--')
    savepath = os.path.join(SAVEDIR, 'dtt_after_train')
    dtt.measure_dtt_ff(network, train_pwl, savepath, verbose=True, exist_ok=True)


    print('--saving distance through layers after training (test)--')
    savepath = os.path.join(SAVEDIR, 'dtt_after_test')
    dtt.measure_dtt_ff(network, test_pwl, savepath, verbose=True, exist_ok=True)

    print('--saving pca after training (train)--')
    savepath = os.path.join(SAVEDIR, 'pca_after_train')
    traj = pca_ff.find_trajectory(network, train_pwl, 2)
    pca_ff.plot_trajectory(traj, savepath, exist_ok=True, alpha=1.0)

    print('--saving pca after training (test)--')
    savepath = os.path.join(SAVEDIR, 'pca_after_test')
    traj = pca_ff.find_trajectory(network, train_pwl, 2)
    pca_ff.plot_trajectory(traj, savepath, exist_ok=True, alpha=1.0)

if __name__ == '__main__':
    main()