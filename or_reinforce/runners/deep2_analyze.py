"""Analyzes a deep2 model"""
import argparse
import os

import shared.setup_torch # pylint: disable=unused-import
import numpy as np
import matplotlib.colors as mcolors

import shared.measures.pca_gen as pca_gen
import shared.measures.participation_ratio as pr
import shared.measures.pca_3d as pca_3d
import shared.filetools as filetools

import or_reinforce.deep.replay_buffer as replay_buffer
import or_reinforce.runners.deep_trainer as deep_trainer
import or_reinforce.deep.deep2 as deep2
import or_reinforce.utils.pca_deep2 as pca_deep2

SAVEDIR = filetools.savepath()
MODULE = 'or_reinforce.runners.deep2_analyze'

def _ots():
    return pca_gen.MaxOTSMapping()

def _markers():
    def markers(inp: np.ndarray):
        argmaxed = inp.argmax(1)
        res = []
        for i, marker in enumerate(['<', '>', '^', 'v']):
            res.append((argmaxed == i, marker))
        return res
    return markers

def _norm():
    return mcolors.Normalize() # autoscale

def main():
    """Main entry point for analyizing the model"""
    parser = argparse.ArgumentParser(description='Trains the deep.deep1 bot against a random bot')
    parser.add_argument('--py3', action='store_true', help='changes executable to python3')
    parser.add_argument('--numthreads', type=int, default=8,
                        help='number of threads to use for gathering experiences')
    parser.add_argument('--train_force_amount', type=float, default=0.1,
                        help='perc moves chosen by chance')
    parser.add_argument('--regul_factor', type=float, default=0.1,
                        help='regularization factor for criterion')
    parser.add_argument('--numexps', type=int, default=1000,
                        help='number of experiences to sample')
    parser.add_argument('--pca3d', action='store_true',
                        help='create the pca3d video')
    parser.add_argument('--mpf', type=float, default=16.67, help='milliseconds per frame')
    args = parser.parse_args()
    _run(args)

def _run(args):
    executable = 'python3' if args.py3 else 'python'
    port = 1769
    nthreads = args.numthreads

    settings = deep_trainer.TrainSettings(
        train_bot='or_reinforce.deep.deep2.deep2',
        adver_bot='optimax_rogue_bots.randombot.RandomBot',
        bot_folder=os.path.join('out', 'or_reinforce', 'deep', 'deep2'),
        train_seq=[
            deep_trainer.SessionSettings(
                tie_len=111, tar_ticks=args.numexps,
                train_force_amount=args.train_force_amount,
                regul_factor=args.regul_factor,
                holdover=0, balance=True, balance_technique='action')
        ],
        cur_ind=0
    )
    deep_trainer._get_experiences_async( # pylint: disable=protected-access
        settings, executable, port, port+nthreads*10, 0,
        False, False, nthreads)

    replay = replay_buffer.FileReadableReplayBuffer(deep2.REPLAY_FOLDER)
    try:
        print(f'loaded {len(replay)} experiences for analysis...')

        network = deep2.Deep2Network.load(deep2.EVAL_MODELFILE)
        pwl = deep2.MyPWL(replay, deep2.Deep2Network.load(deep2.EVAL_MODELFILE))

        print('--fetching top 3 pcs--')
        traj: pca_gen.PCTrajectoryGen = pca_gen.find_trajectory(network, pwl, 3)
        if args.pca3d:
            print('--performing 3d plot--')
            pca_3d.plot_gen(traj, os.path.join(SAVEDIR, 'pca_3d'), True,
                            MODULE + '._markers', MODULE + '._ots', MODULE + '._norm',
                            'cividis', args.mpf, None,
                            ['Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4',
                             'Layer 5', 'Layer 6', 'Output'])
        print('--plotting top 2 pcs--')
        pca_deep2.plot_trajectory(traj, os.path.join(SAVEDIR, 'pca'), exist_ok=True,
                                  transparent=False, norm=mcolors.Normalize(-0.1, 0.1))
        print('--measuring participation ratio--')
        pr_traj: pr.PRTrajectory = pr.measure_pr_gen(network, pwl)
        print('--plotting participation ratio--')
        pr.plot_pr_trajectory(pr_traj, os.path.join(SAVEDIR, 'pr'), exist_ok=True)

        print('--finished--')
    finally:
        replay.close()

if __name__ == '__main__':
    main()
