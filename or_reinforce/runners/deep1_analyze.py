"""Analyzes a deep1 model"""
import argparse
import os

from shared.models.ff import FFTeacher
import shared.measures.pca_gen as pca_gen
import shared.filetools as filetools

import or_reinforce.deep.replay_buffer as replay_buffer
import or_reinforce.runners.deep1 as deep1_runner
import or_reinforce.deep.deep1 as deep1

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
                tie_len=111, tar_ticks=2000,
                train_force_amount=args.train_force_amount)
        ],
        cur_ind=0
    )
    deep1_runner._get_experiences_async( # pylint: disable=protected-access
        settings, executable, port, port+nthreads*10, 0,
        False, False, nthreads)

    replay = replay_buffer.FileReadableReplayBuffer(deep1.REPLAY_FOLDER)
    try:
        print(f'loaded {len(replay)} experiences for analysis...')

        network = deep1.Deep1ModelEval.load(deep1.EVAL_MODELFILE)
        teacher = deep1.MyTeacher(FFTeacher())

        pwl = deep1.MyPWL(replay, deep1.Deep1ModelEval.load(deep1.EVAL_MODELFILE), teacher)

        print('--fetching top 2 pcs--')
        traj: pca_gen.PCTrajectoryGen = pca_gen.find_trajectory(network, pwl, 2)
        print('--plotting top 2 pcs--')
        pca_gen.plot_trajectory(traj, os.path.join(SAVEDIR, 'pca'), exist_ok=True, transparent=False, compress=False,
                                s=16)
        print('--finished--')
    finally:
        replay.close()

if __name__ == '__main__':
    main()
