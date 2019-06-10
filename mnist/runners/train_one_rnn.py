"""Trains a recurrent network on MNIST classification
"""
import shared.setup_torch #pylint: disable=unused-import
from shared.models.rnn import NaturalRNN, RNNTeacher, RNNHiddenActivations
import shared.trainer as tnr
import shared.weight_inits as wi
import shared.measures.dist_through_time as dtt
import shared.measures.pca_ff as pca_ff
import shared.measures.pca as pca
import shared.measures.participation_ratio as pr
import shared.measures.svm as svm
import shared.measures.saturation as satur
import shared.measures.pca3d_throughtrain as pca3d_throughtrain
import shared.filetools
import shared.measures.pca_3d as pca_3d
import shared.npmp as npmp
import shared.criterion as mycrits
import shared.measures.utils as mutils
import torch
from mnist.pwl import MNISTData
import os

SAVEDIR = shared.filetools.savepath()


def main():
    """Entry point"""
    train_pwl = MNISTData.load_train().to_pwl().restrict_to(set(range(10))).rescale()
    test_pwl = MNISTData.load_test().to_pwl().restrict_to(set(range(10))).rescale()
    network = NaturalRNN.create(
        'tanh', train_pwl.input_dim, 200, train_pwl.output_dim,
        input_weights=wi.OrthogonalWeightInitializer(0.03, 0),
        input_biases=wi.ZerosWeightInitializer(), #
        hidden_weights=wi.SompolinskySmoothedFixedGainWeightInitializer(0.001, 20),
        hidden_biases=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=0),
        output_weights=wi.GaussianWeightInitializer(mean=0, vari=0.3, normalize_dim=0),
        output_biases=wi.ZerosWeightInitializer()
    )

    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=RNNTeacher(recurrent_times=10, input_times=1),
        batch_size=30,
        learning_rate=0.003,
        optimizer=torch.optim.RMSprop([p for p in network.parameters() if p.requires_grad], lr=0.003, alpha=0.9),
        criterion=torch.nn.CrossEntropyLoss()
    )

    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(0.1))
     .reg(tnr.InfOrNANDetecter())
     .reg(tnr.InfOrNANDetecter())
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
     .reg(tnr.AccuracyTracker(5, 1000, True))
    )

    print('--saving pcs before training--')
    traj = pca.find_trajectory(network, train_pwl, 10, 2)
    savepath = os.path.join(SAVEDIR, 'pca_before_train')
    pca.plot_trajectory(traj, savepath, exist_ok=True)
    traj = pca.find_trajectory(network, test_pwl, 10, 2)
    savepath = os.path.join(SAVEDIR, 'pca_before_test')
    pca.plot_trajectory(traj, savepath, exist_ok=True)
    del traj

    # print('--saving distance through time before training--')
    # savepath = os.path.join(SAVEDIR, 'dtt_before_train')
    # dtt.measure_dtt(network, train_pwl, 10, savepath, verbose=True, exist_ok=True)
    # savepath = os.path.join(SAVEDIR, 'dtt_before_test')
    # dtt.measure_dtt(network, test_pwl, 10, savepath, verbose=True, exist_ok=True)


    print('--training--')
    result = trainer.train(network)
    print('--finished training--')
    print(result)

    print('--saving pcs after training--')
    traj = pca.find_trajectory(network, train_pwl, 10, 2)
    savepath = os.path.join(SAVEDIR, 'pca_after_train')
    pca.plot_trajectory(traj, savepath, exist_ok=True)
    traj = pca.find_trajectory(network, test_pwl, 10, 2)
    savepath = os.path.join(SAVEDIR, 'pca_after_test')
    pca.plot_trajectory(traj, savepath, exist_ok=True)
    del traj

    # print('--saving distance through time after training--')
    # savepath = os.path.join(SAVEDIR, 'dtt_after_train')
    # dtt.measure_dtt(network, train_pwl, 10, savepath, verbose=True, exist_ok=True)
    # savepath = os.path.join(SAVEDIR, 'dtt_after_test')
    # dtt.measure_dtt(network, test_pwl, 10, savepath, verbose=True, exist_ok=True)

    print('--saving 3d pca plots after training--')
    layer_names = ['Input']
    for i in range(trainer.teacher.recurrent_times + 1):
        layer_names.append(f'Timestep {i+1}')
    dig = npmp.NPDigestor('mnist_train_one_rnn', 2)
    nha = mutils.get_hidacts_rnn(network, train_pwl, trainer.teacher.recurrent_times)
    nha.torch()
    traj = pca_ff.to_trajectory(nha.sample_labels, nha.hid_acts, 3)
    pca_3d.plot_ff(traj, os.path.join(SAVEDIR, 'pca3d_after_train'), False, digestor=dig,
                   frame_time=1000, layer_names=layer_names)

    nha = mutils.get_hidacts_rnn(network, test_pwl, trainer.teacher.recurrent_times)
    nha.torch()
    traj = pca_ff.to_trajectory(nha.sample_labels, nha.hid_acts, 3)
    pca_3d.plot_ff(traj, os.path.join(SAVEDIR, 'pca3d_after_test'), False, digestor=dig,
                   frame_time=1000, layer_names=layer_names)

    dig.join()

if __name__ == '__main__':
    main()