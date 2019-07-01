"""Analyzes a deep1 model"""
import argparse
import os

from shared.models.ff import FFTeacher
import shared.measures.pca_ff as pca_ff
import shared.measures.pca_3d as pca_3d
import shared.filetools as filetools

import or_reinforce.deep.replay_buffer as replay_buffer
import or_reinforce.runners.deep1 as deep1_runner
import or_reinforce.deep.deep1 as deep1
import or_reinforce.utils.general as gen

SAVEDIR = filetools.savepath()

def main():
    """Main entry point for analyizing the model"""
    parser = argparse.ArgumentParser(description='Trains the deep.deep1 bot against a random bot')
    parser.add_argument('--py3', action='store_true', help='changes executable to python3')
    parser.add_argument('--numthreads', type=int, default=8,
                        help='number of threads to use for gathering experiences')
    parser.add_argument('--train_force_amount', type=float, default=0.2,
                        help='perc moves chosen by chance')
    parser.add_argument('--numexps', type=int, default=1000,
                        help='number of experiences to sample')
    args = parser.parse_args()
    _run(args)

def _run(args):
    executable = 'python3' if args.py3 else 'python'
    port = 1769
    nthreads = args.numthreads

    settings = deep1_runner.TrainSettings(
        train_bot='or_reinforce.deep.deep1.deep1',
        adver_bot='optimax_rogue_bots.randombot.RandomBot',
        bot_folder=os.path.join('out', 'or_reinforce', 'deep', 'deep1'),
        train_seq=[
            deep1_runner.SessionSettings(
                tie_len=111, tar_ticks=20000,
                train_force_amount=args.train_force_amount)
        ],
        cur_ind=0
    )
    deep1_runner._get_experiences_async( # pylint: disable=protected-access
        settings, executable, port, nthreads*10, 0,
        False, False, nthreads)

    replay = replay_buffer.FileReadableReplayBuffer(deep1.REPLAY_FOLDER)

    print(f'loaded {len(replay)} experiences for analysis...')

    network = gen.load_model(deep1.MODELFILE)
    teacher = deep1.MyTeacher(FFTeacher())

    pwl = deep1.MyPWL(replay, gen.load_model(deep1.MODELFILE), teacher)

    # todo: discretize the output space for analysis such that we assign
    # a particular label to perhaps 0.9-1, then 0.8-0.9, etc, so we can
    # use our normal classification tools for analysis. This just
    # requires wrapping the PWL. For the technique above, it would literally
    # be labels -> floor(labels * 10)

    # print('--fetching top 3 pcs--')
    # traj: pca_ff.PCTrajectoryFF = pca_ff.find_trajectory(network, pwl, 3)
    # print('--plotting top 2 pcs--')
    # pca_ff.plot_trajectory(traj, os.path.join(SAVEDIR, 'pca'), exist_ok=True)
    # print('--plotting top 3 pcs--')
    # pca_3d.plot_ff(traj, os.path.join(SAVEDIR, 'pca_3d'), exist_ok=True,
    #               layer_names=['Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Output'])
    print('--finished--')

if __name__ == '__main__':
    main()
