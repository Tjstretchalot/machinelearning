"""Trains a single feedforward model on the classifications task
"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.ff import FeedforwardComplex, FFTeacher
from shared.models.ff import ComplexLayer as CL
import shared.trainer as tnr
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import shared.measures.pca_3d as pca_3d
import shared.measures.pca3d_throughtrain as pca3d_throughtrain
import shared.measures.saturation as satur
import shared.measures.participation_ratio as pr
import shared.measures.svm as svm
import shared.criterion as mycrits
import shared.filetools
import shared.npmp as npmp
import shared.convutils as cu
import shared.weight_inits as wi
import torch
import torch.nn

from mnist.pwl import MNISTData
import os


SAVEDIR = shared.filetools.savepath()

INPUT_DIM = 28*28 # not modifiable
HIDDEN_DIM = 250
OUTPUT_DIM = 10

def main():
    """Entry point"""

    nets = cu.FluentShape(28*28).verbose()
    network = FeedforwardComplex(
        INPUT_DIM, OUTPUT_DIM,
        [
            nets.linear_(HIDDEN_DIM),
            nets.tanh(),
            nets.linear_(OUTPUT_DIM),
            nets.tanh()
        ]
    )

    train_pwl = MNISTData.load_train().to_pwl().restrict_to(set(range(10))).rescale()
    test_pwl = MNISTData.load_test().to_pwl().restrict_to(set(range(10))).rescale()

    layer_names = ('Input', 'Hidden', 'Output')

    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=FFTeacher(),
        batch_size=45,
        learning_rate=0.001,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.001),
        criterion=mycrits.create_meansqerr_regul(0.1)#torch.nn.CrossEntropyLoss()
    )

    dig = npmp.NPDigestor('train_one_complex', 35)

    dtt_training_dir = os.path.join(SAVEDIR, 'dtt')
    pca_training_dir = os.path.join(SAVEDIR, 'pca')
    pca3d_training_dir = os.path.join(SAVEDIR, 'pca3d')
    pr_training_dir = os.path.join(SAVEDIR, 'pr')
    svm_training_dir = os.path.join(SAVEDIR, 'svm')
    satur_training_dir = os.path.join(SAVEDIR, 'saturation')
    trained_net_dir = os.path.join(SAVEDIR, 'trained_model')
    pca_throughtrain_dir = os.path.join(SAVEDIR, 'pca_throughtrain')
    logpath = os.path.join(SAVEDIR, 'log.txt')
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(300))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
     .reg(tnr.WeightNoiser(wi.GaussianWeightInitializer(mean=0, vari=0.1), (lambda ctx: ctx.model.layers[0].action.weight.data.detach()), 'add', (lambda noise: wi.GaussianWeightInitializer(0, noise.vari * 0.5))))
     .reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True, dig), skip=1))
     .reg(tnr.OnEpochCaller.create_every(pca_3d.during_training(pca3d_training_dir, True, dig, plot_kwargs={'layer_names': layer_names}), start=1000, skip=1000))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(satur.during_training(satur_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(tnr.save_model(trained_net_dir), skip=100))
     #.reg(pca3d_throughtrain.PCAThroughTrain(pca_throughtrain_dir, layer_names, True, layer_indices=plot_layers))
     .reg(tnr.OnFinishCaller(lambda *args, **kwargs: dig.join()))
     .reg(tnr.CopyLogOnFinish(logpath))
     .reg(tnr.ZipDirOnFinish(dtt_training_dir))
     .reg(tnr.ZipDirOnFinish(pca_training_dir))
     .reg(tnr.ZipDirOnFinish(pca3d_training_dir))
     .reg(tnr.ZipDirOnFinish(pr_training_dir))
     .reg(tnr.ZipDirOnFinish(svm_training_dir))
     .reg(tnr.ZipDirOnFinish(satur_training_dir))
     .reg(tnr.ZipDirOnFinish(trained_net_dir))
    )

    trainer.train(network)
    dig.archive_raw_inputs(os.path.join(SAVEDIR, 'digestor_raw.zip'))

if __name__ == '__main__':
    main()