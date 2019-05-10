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
import shared.measures.pca3d_throughtrain as pca3d_throughtrain
import shared.filetools
import shared.measures.pca_3d as pca_3d
import shared.npmp as npmp
import shared.criterion as mycrits
import torch
from mnist.pwl import MNISTData
import os

SAVEDIR = shared.filetools.savepath()


def main():
    """Entry point"""
    train_pwl = MNISTData.load_train().to_pwl().restrict_to(set(range(10))).rescale()
    test_pwl = MNISTData.load_test().to_pwl().restrict_to(set(range(10))).rescale()

    layers_and_nonlins = (
        (90, 'tanh'),
        (90, 'tanh'),
        (90, 'tanh'),
    )

    layers = [lyr[0] for lyr in layers_and_nonlins]
    nonlins = [lyr[1] for lyr in layers_and_nonlins]
    nonlins.append('linear') # output
    #layer_names = [f'{lyr[1]} (layer {idx})' for idx, lyr in enumerate(layers_and_nonlins)]
    layer_names = [f'Layer {idx+1}' for idx, lyr in enumerate(layers_and_nonlins)]
    layer_names.insert(0, 'input')
    layer_names.append('output')

    network = FeedforwardLarge.create(
        input_dim=train_pwl.input_dim, output_dim=train_pwl.output_dim,
        weights=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=0),
        biases=wi.ZerosWeightInitializer(),
        layer_sizes=layers,
        nonlinearity=nonlins
        #layer_sizes=[500, 200]
    )

    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=FFTeacher(),
        batch_size=30,
        learning_rate=0.003,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.003),
        criterion=mycrits.meansqerr#torch.nn.CrossEntropyLoss()#
    )

    #pca3d_throughtrain.FRAMES_PER_TRAIN = 4
    #pca3d_throughtrain.SKIP_TRAINS = 0
    #pca3d_throughtrain.NUM_FRAME_WORKERS = 6
    pca_3d.DPI = 10 # test
    pca_3d.INPUT_SPIN_TIME = 1000
    pca_3d.OTHER_SPIN_TIME = 1000
    pca_3d.INTERP_SPIN_TIME = 5000

    dig = npmp.NPDigestor('train_one', 35)

    dtt_training_dir = os.path.join(SAVEDIR, 'dtt')
    pca_training_dir = os.path.join(SAVEDIR, 'pca')
    pca3d_training_dir = os.path.join(SAVEDIR, 'pca3d')
    pr_training_dir = os.path.join(SAVEDIR, 'pr')
    svm_training_dir = os.path.join(SAVEDIR, 'svm')
    satur_training_dir = os.path.join(SAVEDIR, 'saturation')
    trained_net_dir = os.path.join(SAVEDIR, 'trained_model')
    pca_throughtrain_dir = os.path.join(SAVEDIR, 'pca_throughtrain')
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(0.2))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
     .reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(pca_3d.during_training(pca3d_training_dir, True, dig, plot_kwargs={'layer_names': layer_names}), start=5, skip=100))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(satur.during_training(satur_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(tnr.save_model(trained_net_dir), skip=100))
     #.reg(pca3d_throughtrain.PCAThroughTrain(pca_throughtrain_dir, layer_names, True))
     .reg(tnr.OnFinishCaller(lambda *args, **kwargs: dig.join()))
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
