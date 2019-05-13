"""Redoes the plots for an already trained model
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
INPFILE = 'trained_model.pt'

def main():
    #train_pwl = MNISTData.load_train().to_pwl().restrict_to(set(range(10))).rescale()
    test_pwl = MNISTData.load_test().to_pwl().restrict_to(set(range(10))).rescale()

    layers_and_nonlins = (
        (90, 'tanh'),
        (90, 'tanh'),
        (90, 'tanh'),
    )

    #layers = [lyr[0] for lyr in layers_and_nonlins]
    nonlins = [lyr[1] for lyr in layers_and_nonlins]
    nonlins.append('linear') # output
    #layer_names = [f'{lyr[1]} (layer {idx})' for idx, lyr in enumerate(layers_and_nonlins)]
    layer_names = [f'Layer {idx+1}' for idx, lyr in enumerate(layers_and_nonlins)]
    layer_names.insert(0, 'input')
    layer_names.append('output')

    network = torch.load(INPFILE)

    dig = npmp.NPDigestor('train_one', 8)
    pca3d_dir = os.path.join(SAVEDIR, 'pca3d')
    traj = pca_ff.find_trajectory(network, test_pwl, 3)
    pca_3d.plot_ff(traj, pca3d_dir, False, 16.67, dig, layer_names=layer_names)

    dig.join()

if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print('failed to set multiprocessing spawn method; this happens on windows')
    main()