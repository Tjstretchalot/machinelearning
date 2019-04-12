"""Trains a single feedforward model on the mnist task
"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.ff import FeedforwardLarge, FFTeacher
import shared.trainer as tnr
import shared.weight_inits as wi
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import shared.measures.participation_ratio as pr
import shared.measures.svm as svm
import shared.measures.saturation as satur
import shared.filetools
import shared.measures.pca_3d as pca_3d
import shared.npmp as npmp
import torch
from mnist.pwl import MNISTData
import os

SAVEDIR = shared.filetools.savepath()

def _meansqerr_criterion(output: torch.tensor, labels: torch.tensor):
    adj_output = output.clone()
    adj_output[labels] -= 1
    return torch.mean(adj_output ** 2)

def main():
    """Entry point"""
    train_pwl = MNISTData.load_train().to_pwl().restrict_to(set(range(10))).rescale()
    test_pwl = MNISTData.load_test().to_pwl().restrict_to(set(range(10))).rescale()

    network = FeedforwardLarge.create(
        input_dim=train_pwl.input_dim, output_dim=train_pwl.output_dim, nonlinearity='tanh',
        weights=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=1),
        biases=wi.ZerosWeightInitializer(),
        layer_sizes=[90, 90, 90, 90, 90, 25]
        #layer_sizes=[500, 200]
    )

    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=FFTeacher(),
        batch_size=30,
        learning_rate=0.001,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.003),
        criterion=_meansqerr_criterion#torch.nn.CrossEntropyLoss()
    )

    dig = npmp.NPDigestor('train_one', 35)

    dtt_training_dir = os.path.join(SAVEDIR, 'dtt')
    pca_training_dir = os.path.join(SAVEDIR, 'pca')
    pca3d_training_dir = os.path.join(SAVEDIR, 'pca3d')
    pr_training_dir = os.path.join(SAVEDIR, 'pr')
    svm_training_dir = os.path.join(SAVEDIR, 'svm')
    satur_training_dir = os.path.join(SAVEDIR, 'saturation')
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(150))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
     .reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True, dig), skip=1000))
     #.reg(tnr.OnEpochCaller.create_every(pca_3d.during_training(pca3d_training_dir, True, dig), skip=1, stop=15))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True, dig), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True, dig), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True, dig), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(satur.during_training(satur_training_dir, True, dig), skip=1000))
     .reg(tnr.OnFinishCaller(lambda *args, **kwargs: dig.join()))
     .reg(tnr.ZipDirOnFinish(dtt_training_dir))
     .reg(tnr.ZipDirOnFinish(pca_training_dir))
     #.reg(tnr.ZipDirOnFinish(pca3d_training_dir))
     .reg(tnr.ZipDirOnFinish(pr_training_dir))
     .reg(tnr.ZipDirOnFinish(svm_training_dir))
     .reg(tnr.ZipDirOnFinish(satur_training_dir))
    )

    trainer.train(network)
    dig.archive_raw_inputs(os.path.join(SAVEDIR, 'digestor_raw.zip'))

if __name__ == '__main__':
    main()
