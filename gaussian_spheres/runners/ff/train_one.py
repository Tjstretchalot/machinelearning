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
     .reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True), skip=10))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True), skip=10))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True), skip=10))
     .reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True), skip=10))
     .reg(tnr.ZipDirOnFinish(dtt_training_dir))
     .reg(tnr.ZipDirOnFinish(pca_training_dir))
     .reg(tnr.ZipDirOnFinish(pr_training_dir))
     .reg(tnr.ZipDirOnFinish(svm_training_dir))
    )
    trainer.train(network)

if __name__ == '__main__':
    main()