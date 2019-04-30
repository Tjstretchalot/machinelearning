"""Trains a single two-layer feedforward model without bias on
a binary classification task.
"""

import shared.setup_torch #pylint: disable=unused-import
from shared.models.ff import FeedforwardLarge, FFTeacher
import shared.trainer as tnr
import shared.weight_inits as wi
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import shared.measures.pca_3d as pca_3d
import shared.measures.pca3d_throughtrain as pca3d_throughtrain
import shared.measures.participation_ratio as pr
import shared.measures.svm as svm
import shared.measures.saturation as satur
import shared.filetools
import shared.npmp as npmp
import shared.criterion as mycrits
import torch
from shared.pwl import PointWithLabel
from gaussian_spheres.pwl import GaussianSpheresPWLP
import os


SAVEDIR = shared.filetools.savepath()

def main():
    """Entry point"""
    pwl = GaussianSpheresPWLP(
        epoch_size=100,
        input_dim=3,
        output_dim=3,
        clusters=[PointWithLabel(point=torch.tensor((1, 0, 0), dtype=torch.double), label=0),
                  PointWithLabel(point=torch.tensor((0, 1, 0), dtype=torch.double), label=1),
                  PointWithLabel(point=torch.tensor((0, 0, 1), dtype=torch.double), label=2)],
        std_dev=0.5,
        mean=0
    )

    layers = [(50, True, False)]
    layer_names = ['input', 'hidden', 'output']

    network = FeedforwardLarge.create(
        input_dim=3, output_dim=3,
        weights=wi.GaussianWeightInitializer(mean=0.2, vari=0.1, normalize_dim=1),
        biases=wi.ZerosWeightInitializer(),
        layer_sizes=layers,
        nonlinearity='tanh',
        train_readout_weights=False,
        train_readout_bias=False
    )

    trainer = tnr.GenericTrainer(
        train_pwl=pwl,
        test_pwl=pwl,
        teacher=FFTeacher(),
        batch_size=1,
        learning_rate=0.003,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.003),
        criterion=mycrits.create_meansqerr_regul(noise_strength=0.1)#torch.nn.CrossEntropyLoss()
    )

    pca3d_throughtrain.FRAMES_PER_TRAIN = 1
    pca3d_throughtrain.SKIP_TRAINS = 0
    pca3d_throughtrain.NUM_FRAME_WORKERS = 4

    dig = npmp.NPDigestor('train_one', 5)
    #pca_3d.plot_ff(pca_ff.find_trajectory(network, pwl, 3), os.path.join(SAVEDIR, 'pca_3d_start'), True,
    #               digestor=dig, frame_time=FRAME_TIME, layer_names=layer_names)
    dtt_training_dir = os.path.join(SAVEDIR, 'dtt')
    pca_training_dir = os.path.join(SAVEDIR, 'pca')
    pr_training_dir = os.path.join(SAVEDIR, 'pr')
    svm_training_dir = os.path.join(SAVEDIR, 'svm')
    satur_training_dir = os.path.join(SAVEDIR, 'saturation')
    pca_throughtrain_dir = os.path.join(SAVEDIR, 'pca_throughtrain')
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(100))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
     #.reg(tnr.WeightNoiser(
     #    wi.GaussianWeightInitializer(mean=0, vari=0.02, normalize_dim=None),
     #    lambda ctxt: ctxt.model.layers[-1].weight.data))
     .reg(tnr.OnEpochCaller.create_every(satur.during_training(satur_training_dir, True, dig), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True, dig), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True, dig), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True, dig), skip=1000))
     .reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True, dig), skip=1000))
     .reg(pca3d_throughtrain.PCAThroughTrain(pca_throughtrain_dir, layer_names, True))
     .reg(tnr.OnFinishCaller(lambda *args, **kwargs: dig.join()))
     .reg(tnr.ZipDirOnFinish(dtt_training_dir))
     .reg(tnr.ZipDirOnFinish(pca_training_dir))
     .reg(tnr.ZipDirOnFinish(pr_training_dir))
     .reg(tnr.ZipDirOnFinish(svm_training_dir))
     .reg(tnr.ZipDirOnFinish(satur_training_dir))
    )
    trainer.train(network)
    #pca_3d.plot_ff(pca_ff.find_trajectory(network, pwl, 3), os.path.join(SAVEDIR, 'pca_3d_end'), True,
    #               digestor=dig, frame_time=FRAME_TIME, layer_names=layer_names)
    dig.archive_raw_inputs(os.path.join(SAVEDIR, 'raw_digestor.zip'))

if __name__ == '__main__':
    main()