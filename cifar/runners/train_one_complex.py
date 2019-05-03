"""Trains a single feedforward model on the classifications task
"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.ff import FeedforwardComplex, FFTeacher
import shared.weight_inits as wi
import shared.trainer as tnr
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import shared.measures.pca_3d as pca_3d
import shared.measures.pca3d_throughtrain as pca3d_throughtrain
import shared.measures.saturation as satur
import shared.measures.participation_ratio as pr
import shared.measures.svm as svm
import shared.filetools
import shared.npmp as npmp
import shared.convutils as cu
import torch
import torch.nn

from cifar.pwl import CIFARData
import os


SAVEDIR = shared.filetools.savepath()

INPUT_DIM = 32*32*3 # not modifiable
OUTPUT_DIM = 10

STOP_EPOCH = 200

            # nets.unflatten_conv3_(1, 3, 32, 32),
            # nets.conv3_(16, (3, 5, 5), stride=(3, 1, 1), padding=(0, 2, 2)),
            # nets.relu(),
            # nets.maxpool3_((1, 2, 2), stride=(1, 2, 2)),
            # nets.conv3_(20, (1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
            # nets.relu(),
            # nets.maxpool3_((1, 2, 2), stride=(1, 2, 2)),
            # nets.conv3_(20, (1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
            # nets.relu(),
            # nets.maxpool3_((1, 2, 2), stride=(1, 2, 2)),
            # nets.flatten_(invokes_callback=True),
            # nets.linear_(OUTPUT_DIM),
    #layer_names = ('input', 'conv3d-relu', 'conv3-relu', 'conv3d-relu', 'output')
    #plot_layers = (3,)
def main():
    """Entry point"""

    cu.DEFAULT_LINEAR_BIAS_INIT = wi.ZerosWeightInitializer()
    cu.DEFAULT_LINEAR_WEIGHT_INIT = wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=0)

    nets = cu.FluentShape(32*32*3).verbose()
    network = FeedforwardComplex(
        INPUT_DIM, OUTPUT_DIM,
        [
            nets.linear_(32*32*6),
            nets.nonlin('isrlu'),
            nets.linear_(500),
            nets.nonlin('isrlu'),
            nets.linear_(250),
            nets.nonlin('isrlu'),
            nets.linear_(250),
            nets.nonlin('isrlu'),
            nets.linear_(100),
            nets.tanh(),
            nets.linear_(100),
            nets.tanh(),
            nets.linear_(100),
            nets.tanh(),
            nets.linear_(OUTPUT_DIM),
            nets.nonlin('isrlu'),
        ]
    )

    train_pwl = CIFARData.load_train().to_pwl().restrict_to(set(range(10))).rescale()
    test_pwl = CIFARData.load_test().to_pwl().restrict_to(set(range(10))).rescale()

    layer_names = ('input',
                   'FC -> 32*32*6 (ISRLU)', 'FC -> 500 (ISRLU)',
                   'FC -> 250 (ISRLU)', 'FC -> 250 (ISRLU)',
                   'FC -> 100 (tanh)', 'FC -> 100 (tanh)',
                   'FC -> 100 (tanh)',
                   f'FC -> {OUTPUT_DIM} (isrelu)')
    plot_layers = tuple(i for i in range(2, len(layer_names) - 1))
    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=FFTeacher(),
        batch_size=45,
        learning_rate=0.001,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.001),
        criterion=torch.nn.CrossEntropyLoss()
    )

    pca3d_throughtrain.FRAMES_PER_TRAIN = 1
    pca3d_throughtrain.SKIP_TRAINS = 16
    pca3d_throughtrain.NUM_FRAME_WORKERS = 1

    dig = npmp.NPDigestor('train_one_complex', 5)

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
     .reg(tnr.EpochsStopper(STOP_EPOCH))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.EpochProgress(print_every=120, hint_end_epoch=STOP_EPOCH))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau(patience=3))
     .reg(tnr.AccuracyTracker(1, 1000, True))
     .reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True, dig), skip=5))
     .reg(tnr.OnEpochCaller.create_every(pca_3d.during_training(pca3d_training_dir, True, dig, plot_kwargs={'layer_names': layer_names}), start=10, skip=100))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True, dig), skip=5))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True, dig, labels=False), skip=5))
     .reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True, dig), skip=5))
     .reg(tnr.OnEpochCaller.create_every(satur.during_training(satur_training_dir, True, dig), skip=5))
     .reg(tnr.OnEpochCaller.create_every(tnr.save_model(trained_net_dir), skip=5))
     #.reg(pca3d_throughtrain.PCAThroughTrain(pca_throughtrain_dir, layer_names, True, layer_indices=plot_layers))
     .reg(tnr.OnFinishCaller(lambda *args, **kwargs: dig.join()))
     .reg(tnr.ZipDirOnFinish(dtt_training_dir))
     .reg(tnr.ZipDirOnFinish(pca_training_dir))
     .reg(tnr.ZipDirOnFinish(pca3d_training_dir))
     .reg(tnr.ZipDirOnFinish(pr_training_dir))
     .reg(tnr.ZipDirOnFinish(svm_training_dir))
     .reg(tnr.ZipDirOnFinish(satur_training_dir))
     .reg(tnr.ZipDirOnFinish(trained_net_dir))
     .reg(tnr.CopyLogOnFinish(logpath))
    )

    trainer.train(network)
    dig.archive_raw_inputs(os.path.join(SAVEDIR, 'digestor_raw.zip'))

if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print('failed to set multiprocessing spawn method; this happens on windows')

    main()