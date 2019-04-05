"""Trains a single feedforward model on the classifications task
"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.ff import FeedforwardLarge, FFTeacher
import shared.trainer as tnr
import shared.weight_inits as wi
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import shared.measures.participation_ratio as pr
import shared.measures.svm as svm
import shared.filetools
import torch
from gaussian_spheres.pwl import GaussianSpheresPWLP
import os


SAVEDIR = shared.filetools.savepath()

INPUT_DIM = 200
OUTPUT_DIM = 3

def main():
    """Entry point"""
    pwl = GaussianSpheresPWLP.create(
        epoch_size=2700, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, cube_half_side_len=2,
        num_clusters=90, std_dev=0.04, mean=0, min_sep=0.1
    )

    network = FeedforwardLarge.create(
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, nonlinearity='tanh',
        weights=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=1),
        biases=wi.ZerosWeightInitializer(),
        layer_sizes=[200, 100, 100, 100, 100, 50, 50]
    )

    trainer = tnr.GenericTrainer(
        train_pwl=pwl,
        test_pwl=pwl,
        teacher=FFTeacher(),
        batch_size=45,
        learning_rate=0.001,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.001),
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
    savepath = os.path.join(SAVEDIR, 'dtt_before')
    dtt.measure_dtt_ff(network, pwl, savepath, verbose=True, exist_ok=True)

    print('--saving pca before training--')
    savepath = os.path.join(SAVEDIR, 'pca_before')
    traj = pca_ff.find_trajectory(network, pwl, 2)
    pca_ff.plot_trajectory(traj, savepath, exist_ok=True)
    del traj

    print('--saving pr before training--')
    savepath = os.path.join(SAVEDIR, 'pr_before')
    traj = pr.measure_pr_ff(network, pwl)
    pr.plot_pr_trajectory(traj, savepath, exist_ok=True)
    del traj

    print('--saving svm traj before training--')
    savepath = os.path.join(SAVEDIR, 'svm_before')
    traj = svm.train_svms_ff(network, pwl)
    svm.plot_traj_ff(traj, savepath, exist_ok=True)
    del traj

    trainer.train(network)

    print('--saving distance through layers after training--')
    savepath = os.path.join(SAVEDIR, 'dtt_after')
    dtt.measure_dtt_ff(network, pwl, savepath, verbose=True, exist_ok=True)


    print('--saving pca after training--')
    savepath = os.path.join(SAVEDIR, 'pca_after')
    traj = pca_ff.find_trajectory(network, pwl, 2)
    pca_ff.plot_trajectory(traj, savepath, exist_ok=True)

    print('--saving pr after training--')
    savepath = os.path.join(SAVEDIR, 'pr_after')
    traj = pr.measure_pr_ff(network, pwl)
    pr.plot_pr_trajectory(traj, savepath, exist_ok=True)

    print('--saving svm traj after training--')
    savepath = os.path.join(SAVEDIR, 'svm_after')
    traj = svm.train_svms_ff(network, pwl)
    svm.plot_traj_ff(traj, savepath, exist_ok=True)
    del traj

if __name__ == '__main__':
    main()