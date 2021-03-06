"""Trains a single feedforward model on the classifications task
with an ODE block in the middle
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
import shared.measures.saturation as satur
import shared.convutils as cu
import shared.filetools
import shared.npmp as npmp
import torch
import torch.nn
from functools import reduce
import operator

from gaussian_spheres.pwl import GaussianSpheresPWLP
import os
import sys


SAVEDIR = shared.filetools.savepath()

INPUT_DIM = 200
ODE_DIM = 90
OUTPUT_DIM = 3

def main():
    """Entry point"""
    pwl = GaussianSpheresPWLP.create(
        epoch_size=2700, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, cube_half_side_len=2,
        num_clusters=10, std_dev=0.4, mean=0, min_sep=0.4
    )

    nets = cu.FluentShape(INPUT_DIM).verbose()
    nets_ode = cu.FluentShape(ODE_DIM)
    network = FeedforwardComplex(INPUT_DIM, OUTPUT_DIM,
        [
            nets.linear_(ODE_DIM),
            nets.nonlin('tanh'),
            nets.ode_(FeedforwardComplex(ODE_DIM, ODE_DIM, [
                nets_ode.linear_(ODE_DIM),
                nets_ode.nonlin('tanh', invokes_callback=False),
            ])),
            nets.linear_(OUTPUT_DIM),
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
    satur_training_dir = os.path.join(SAVEDIR, 'saturation')
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(20))
     .reg(tnr.InfOrNANDetecter())
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(3))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
     #.reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True, alpha=0.8, s=4, transparent=False), skip=1))
     #.reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True), skip=1000))
     #.reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True), skip=1000))
     #.reg(tnr.OnEpochCaller.create_every(satur.during_training(satur_training_dir, True), skip=1000))
     .reg(tnr.ZipDirOnFinish(dtt_training_dir))
     .reg(tnr.ZipDirOnFinish(pca_training_dir))
     .reg(tnr.ZipDirOnFinish(pr_training_dir))
     .reg(tnr.ZipDirOnFinish(svm_training_dir))
     .reg(tnr.ZipDirOnFinish(satur_training_dir))
    )
    trainer.train(network)
    torch.save(network.state_dict(), os.path.join(SAVEDIR, 'trained_network.pt'))
    #pca_3d.plot_ff(pca_ff.find_trajectory(network, pwl, 3), os.path.join(SAVEDIR, 'pca_3d_end'), True, dig3d)
    #dig3d.archive_raw_inputs(os.path.join(SAVEDIR, 'pca_3d_raw.zip'))

if __name__ == '__main__':
   should_clean = '--clean' in sys.argv
   print(f'should_clean = {should_clean} ; sys.argv = {sys.argv}')
   if should_clean:
      import shutil
      shutil.rmtree(SAVEDIR)
   main()