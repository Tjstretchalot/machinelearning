"""Trains a single feedforward model on the classifications task
"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.ff import FeedforwardComplex, FFTeacher
from shared.models.ff import ComplexLayer as CL
import shared.trainer as tnr
import shared.weight_inits as wi
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import shared.measures.pca_3d as pca_3d
import shared.measures.participation_ratio as pr
import shared.measures.svm as svm
import shared.filetools
import shared.npmp as npmp
import torch
import torch.nn
from functools import reduce
import operator

from gaussian_spheres.pwl import GaussianSpheresPWLP
import os


SAVEDIR = shared.filetools.savepath()

INPUT_DIM = 28*28 # not modifiable
OUTPUT_DIM = 3

def main():
    """Entry point"""
    pwl = GaussianSpheresPWLP.create(
        epoch_size=2700, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, cube_half_side_len=2,
        num_clusters=90, std_dev=0.04, mean=0, min_sep=0.1
    )

    network = FeedforwardComplex(INPUT_DIM, OUTPUT_DIM,
        [
            CL(style='other', is_module=False, invokes_callback=False,
               action=lambda x: x.reshape(-1, 1, 28, 28)),
            CL(style='layer', is_module=True, invokes_callback=False,
               action=torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)),
            CL(style='nonlinearity', is_module=False, invokes_callback=True,
               action=torch.relu),
            CL(style='layer', is_module=True, invokes_callback=True,
               action=torch.nn.MaxPool2d(kernel_size=2, stride=1)),
            CL(style='other', is_module=False, invokes_callback=False,
               action=lambda x: x.reshape(-1, reduce(operator.mul, x.shape[1:]))),
            CL(style='layer', is_module=True, invokes_callback=False,
               action=torch.nn.Linear(2645, 200)),
            CL(style='nonlinearity', is_module=False, invokes_callback=True,
               action=torch.tanh),
            CL(style='layer', is_module=True, invokes_callback=False,
               action=torch.nn.Linear(200, OUTPUT_DIM)),
            CL(style='nonlinearity', is_module=False, invokes_callback=True,
               action=torch.tanh)
        ]
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

    dig = npmp.NPDigestor('train_one_complex', 16)
    #pca_3d.plot_ff(pca_ff.find_trajectory(network, pwl, 3), os.path.join(SAVEDIR, 'pca_3d_start'), True, dig3d)
    #dig3d.join()
    #exit()
    dtt_training_dir = os.path.join(SAVEDIR, 'dtt')
    pca_training_dir = os.path.join(SAVEDIR, 'pca')
    pr_training_dir = os.path.join(SAVEDIR, 'pr')
    svm_training_dir = os.path.join(SAVEDIR, 'svm')
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(150))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
     .reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True), skip=1000))
     .reg(tnr.ZipDirOnFinish(dtt_training_dir))
     .reg(tnr.ZipDirOnFinish(pca_training_dir))
     .reg(tnr.ZipDirOnFinish(pr_training_dir))
     .reg(tnr.ZipDirOnFinish(svm_training_dir))
    )
    trainer.train(network)
    torch.save(network.state_dict(), os.path.join(SAVEDIR, 'trained_network.pt'))
    #pca_3d.plot_ff(pca_ff.find_trajectory(network, pwl, 3), os.path.join(SAVEDIR, 'pca_3d_end'), True, dig3d)
    #dig3d.archive_raw_inputs(os.path.join(SAVEDIR, 'pca_3d_raw.zip'))

if __name__ == '__main__':
    main()