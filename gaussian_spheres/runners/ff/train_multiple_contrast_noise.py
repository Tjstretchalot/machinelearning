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
import numpy as np
from gaussian_spheres.pwl import GaussianSpheresPWLP
import os

SAVEDIR = shared.filetools.savepath()
DIM = 64

def train_with_noise(vari, rep, ignoreme): # pylint: disable=unused-argument
    """Entry point"""
    train_pwl = GaussianSpheresPWLP.create(
        epoch_size=2700, input_dim=DIM, output_dim=10, cube_half_side_len=2,
        num_clusters=10, std_dev=0.02, mean=0, min_sep=1, force_split=True
    )
    test_pwl = train_pwl

    layers_and_nonlins = (
        (DIM, 'relu'),
        (DIM, 'relu'),
        (DIM, 'relu'),
        (DIM, 'relu'),
        (DIM, 'relu'),
        (DIM, 'relu'),
        (DIM, 'relu'),
    )

    layers = [lyr[0] for lyr in layers_and_nonlins]
    nonlins = [lyr[1] for lyr in layers_and_nonlins]
    nonlins.append('relu') # output
    #layer_names = [f'{lyr[1]} (layer {idx})' for idx, lyr in enumerate(layers_and_nonlins)]
    layer_names = [f'Layer {idx+1}' for idx, lyr in enumerate(layers_and_nonlins)]
    layer_names.insert(0, 'Input')
    layer_names.append('Output')

    network = FeedforwardLarge.create(
        input_dim=train_pwl.input_dim, output_dim=train_pwl.output_dim,
        weights=wi.GaussianWeightInitializer(mean=0, vari=0.001, normalize_dim=0),
        biases=wi.ZerosWeightInitializer(),
        layer_sizes=layers,
        nonlinearity=nonlins
    )

    _lr = 0.001
    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=FFTeacher(),
        batch_size=30,
        learning_rate=_lr,
        optimizer=torch.optim.SGD([p for p in network.parameters() if p.requires_grad], lr=_lr),#torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.003),
        criterion=mycrits.meansqerr#torch.nn.CrossEntropyLoss()#
    )

    #pca3d_throughtrain.FRAMES_PER_TRAIN = 4
    #pca3d_throughtrain.SKIP_TRAINS = 0
    #pca3d_throughtrain.NUM_FRAME_WORKERS = 6

    dig = npmp.NPDigestor(f'TRMCN_{rep}_{vari}', 8)

    savedir = os.path.join(SAVEDIR, f'variance_{vari}', f'repeat_{rep}')

    dtt_training_dir = os.path.join(savedir, 'dtt')
    pca_training_dir = os.path.join(savedir, 'pca')
    pca3d_training_dir = os.path.join(savedir, 'pca3d')
    pr_training_dir = os.path.join(savedir, 'pr')
    svm_training_dir = os.path.join(savedir, 'svm')
    satur_training_dir = os.path.join(savedir, 'saturation')
    trained_net_dir = os.path.join(savedir, 'trained_model')
    pca_throughtrain_dir = os.path.join(savedir, 'pca_throughtrain')
    logpath = os.path.join(savedir, 'log.txt')
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(25))
     .reg(tnr.DecayTracker())
     #.reg(tnr.DecayStopper(5))
     .reg(tnr.LRMultiplicativeDecayer())
     #.reg(tnr.DecayOnPlateau())
     .reg(tnr.DecayEvery(5))
     .reg(tnr.AccuracyTracker(1, 1000, True))
     .reg(tnr.WeightNoiser(wi.GaussianWeightInitializer(mean=0, vari=vari), (lambda ctx: ctx.model.layers[-1].weight.data.detach()), 'scale', (lambda noise: wi.GaussianWeightInitializer(0, noise.vari * 0.5))))
     #.reg(tnr.OnEpochCaller.create_every(dtt.during_training_ff(dtt_training_dir, True, dig), skip=100))
     #.reg(tnr.OnEpochCaller.create_every(pca_3d.during_training(pca3d_training_dir, True, dig, plot_kwargs={'layer_names': layer_names}), start=500, skip=100))
     #.reg(tnr.OnEpochCaller.create_every(pca_ff.during_training(pca_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(pr.during_training_ff(pr_training_dir, True, dig), skip=1))
     #.reg(tnr.OnEpochCaller.create_every(svm.during_training_ff(svm_training_dir, True, dig), skip=100))
     #.reg(tnr.OnEpochCaller.create_every(satur.during_training(satur_training_dir, True, dig), skip=100))
     .reg(tnr.OnEpochCaller.create_every(tnr.save_model(trained_net_dir), skip=100))
     #.reg(pca3d_throughtrain.PCAThroughTrain(pca_throughtrain_dir, layer_names, True))
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
    dig.archive_raw_inputs(os.path.join(savedir, 'digestor_raw.zip'))

def plot_pr_together(variances, num_repeats=1, fname_hint='pr_epoch_finished', suppress_zip=False):
    """Plots all the data from the given epoch together"""
    trajs_with_meta = []
    for vari in variances:
        repeats = []
        for rep in range(num_repeats):
            savedir = os.path.join(SAVEDIR, f'variance_{vari}', f'repeat_{rep}')
            pr_dir = os.path.join(savedir, 'pr')
            if os.path.exists(pr_dir + '.zip'):
                shared.filetools.unzip(pr_dir + '.zip')

            epoch_dir = os.path.join(pr_dir, fname_hint)
            if os.path.exists(epoch_dir + '.zip'):
                shared.filetools.unzip(epoch_dir + '.zip')

            traj = pr.PRTrajectory.load(os.path.join(epoch_dir, 'traj.zip'))
            if not suppress_zip:
                shared.filetools.zipdir(epoch_dir)
                shared.filetools.zipdir(pr_dir)

            repeats.append(traj)
        trajs_with_meta.append(pr.TrajectoryWithMeta(trajectory=pr.AveragedPRTrajectory(repeats), label=f'$\sigma^2 = {vari}$'))

    savepath = os.path.join(SAVEDIR, f'prs_{fname_hint}')
    print(f'Saving to {savepath} (fname_hint={fname_hint})')
    pr.plot_avg_pr_trajectories(
        trajs_with_meta, savepath, 'PR varying $\sigma^2$', exist_ok=True)

def train(variances, reuse_repeats, num_repeats):
    """Trains all the networks"""
    dig = npmp.NPDigestor('train_mult_contr_noise', 4, target_module='mnist.runners.train_multiple_contrast_noise', target_name='train_with_noise')
    empty_arr = np.array([])
    for vari in variances:
        for i in range(reuse_repeats, num_repeats):
            dig(vari, i, empty_arr)
    dig.join()

def plot_merged(variances, num_repeats):
    """Finds all the epochs that we went through on all of them and plots them"""
    avail_data = None
    for vari in variances:
        for rep in range(num_repeats):
            savedir = os.path.join(SAVEDIR, f'variance_{vari}', f'repeat_{rep}')
            pr_dir = os.path.join(savedir, 'pr')
            if os.path.exists(pr_dir + '.zip'):
                shared.filetools.unzip(pr_dir + '.zip')

            with os.scandir(pr_dir) as files:
                first = avail_data is None
                if first:
                    avail_data = set()
                    missing_data = set()
                else:
                    missing_data = avail_data.copy()

                for item in files:
                    epoch = os.path.splitext(item.name)[0]
                    if first:
                        avail_data.add(epoch)
                    else:
                        missing_data.remove(epoch)
                avail_data -= missing_data

    for avail in avail_data:
        plot_pr_together(variances, num_repeats=num_repeats, fname_hint=avail, suppress_zip=True)

    for vari in variances:
        for rep in range(num_repeats):
            savedir = os.path.join(SAVEDIR, f'variance_{vari}', f'repeat_{rep}')
            pr_dir = os.path.join(savedir, 'pr')
            if not os.path.exists(pr_dir + '.zip'):
                shared.filetools.zipdir(pr_dir)

def main():
    """Main function"""
    #variances = [0, 0.07, 0.14, 0.2]
    #num_repeats = 10
    variances = [0, 0.075, 0.15]
    num_repeats = 2
    reuse_repeats = 0
    train(variances, reuse_repeats, num_repeats)
    plot_merged(variances, num_repeats)

if __name__ == '__main__':
    main()
