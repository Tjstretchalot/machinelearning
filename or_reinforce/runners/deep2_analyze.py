"""Analyzes a deep2 model"""
import argparse
import os
import typing
import json

import shared.setup_torch # pylint: disable=unused-import
import torch
import numpy as np
import matplotlib.colors as mcolors

import shared.measures.pca_gen as pca_gen
import shared.measures.participation_ratio as pr
import shared.measures.pca_3d as pca_3d
import shared.filetools as filetools
import shared.pwl as pwl

import or_reinforce.deep.replay_buffer as replay_buffer
import or_reinforce.runners.deep_trainer as deep_trainer
import or_reinforce.deep.deep2 as deep2
import or_reinforce.utils.pca_deep2 as pca_deep2

import optimax_rogue.logic.moves as moves
from optimax_rogue.game.state import GameState


SAVEDIR = filetools.savepath()
MODULE = 'or_reinforce.runners.deep2_analyze'
MARKERS = ['<', '>', '^', 'v']
USE_MARKER_SHAPES = False
"""True means we use color to signify expected reward and the marker
shape to indicate which move. False to use color to indicate which
move and to use the same marker shape for everything. Marker shape
can be hard to distinguish"""

def _ots():
    return pca_gen.MaxOTSMapping()

def _ots_argmax():
    return pca_gen.ArgmaxOTSMapping()

def _markers():
    def markers(inp: np.ndarray):
        argmaxed = inp.argmax(1)
        res = []
        for i, marker in enumerate(MARKERS):
            res.append((argmaxed == i, marker))
        return res
    return markers

def _all_same_markers():
    def markers(inp: np.ndarray):
        return [(np.ones(inp.shape[0], dtype='bool'), 'o')]
    return markers

STORED_MARKER_FP = os.path.join(SAVEDIR, 'cached_markers')
def _mark_cached_moves():
    def markers(inp: np.ndarray):
        metafile = os.path.join(STORED_MARKER_FP, 'meta.json')
        with open(metafile, 'r') as infile:
            meta = json.load(infile)

        marks = []
        with np.load(os.path.join(STORED_MARKER_FP, 'masks.npz')) as masks:
            for i, marker in enumerate(meta['markers']):
                mask = masks[f'mask_{i}']
                marks.append((mask, marker))
        return marks
    return markers

def _cache_markers(markers: typing.List[typing.Tuple[np.ndarray, str]]):
    """Stores the given mask and marker combination so that it will be loaded
    by _mark_cached_moves and returned"""
    os.makedirs(STORED_MARKER_FP, exist_ok=True)

    metafile = os.path.join(STORED_MARKER_FP, 'meta.json')
    with open(metafile, 'w') as outfile:
        json.dump({
            'markers': list(mark for _, mark in markers)
        }, outfile)

    np.savez_compressed(
        os.path.join(STORED_MARKER_FP, 'masks.npz'),
        **dict((f'mask_{i}', mask) for i, (_, mask) in enumerate(markers)))

def _get_correct(exp: replay_buffer.Experience):
    state: GameState = exp.state
    bot_iden = state.player_1_iden if exp.player_id == 1 else state.player_2_iden
    oth_iden = state.player_1_iden if exp.player_id == 2 else state.player_2_iden
    bot = state.iden_lookup[bot_iden]
    if oth_iden in state.iden_lookup:
        oth = state.iden_lookup[oth_iden]
        if bot.depth == oth.depth and (
                min(abs(bot.y - oth.y), abs(bot.x - oth.x)) <= deep2.ENTITY_VIEW_DIST):
            return deep2.MOVE_MAP
    scase = state.world.get_at_depth(bot.depth).staircase()

    res = []
    if scase[0] < bot.x:
        res.append(moves.Move.Left)
    elif scase[0] > bot.x:
        res.append(moves.Move.Right)
    if scase[1] < bot.y:
        res.append(moves.Move.Up)
    elif scase[1] > bot.y:
        res.append(moves.Move.Down)
    return res

def _is_correct(network: deep2.Deep2Network, exp: replay_buffer.Experience,
                net_res: torch.tensor):
    # net_res = network(torch.from_numpy(exp.encoded_state).unsqueeze(0))
    net_action = deep2.MOVE_MAP[net_res.squeeze().argmax().item()]
    return net_action in _get_correct(exp)

def _correctness_markers(network: deep2.Deep2Network, states: torch.tensor,
                         exps: typing.List[replay_buffer.Experience]):
    """Returns markers that can be passed to _cache_markers that correspond to
    a circle if the network makes the right decision and false otherwise.

    Args:
        network (deep2.Deep2Network): the network who is judging the states
        exps (typing.List[Experience]): the experiences
    """
    net_outs = network(states)

    mask_cor = np.zeros(states.shape[0], dtype='bool')
    for i, (nout, exp) in enumerate(zip(net_outs, exps)):
        if _is_correct(network, exp, nout):
            mask_cor[i] = 1

    mask_incor = 1 - mask_cor

    return [(mask_cor, 'o'), (mask_incor, 'X')]


def _norm():
    return mcolors.Normalize() # autoscale

def _nonorm():
    return mcolors.NoNorm()

def get_unique_states(replay_path: str) -> torch.tensor:
    """Gets the unique encoded states that are in the given replay folder.

    Arguments:
        replay_play (str): the path to where the replay experiences are stored

    Returns:
        unique_states (torch.tensor): the unique encoded game states within
            the experiences
    """
    result = []
    buffer = replay_buffer.FileReadableReplayBuffer(replay_path)
    try:
        for _ in range(len(buffer)):
            exp: replay_buffer.Experience = next(buffer)
            if all((existing != exp.encoded_state).sum() > 0 for existing in result):
                result.append(torch.from_numpy(exp.encoded_state))
    finally:
        buffer.close()

    return torch.cat(tuple(i.unsqueeze(0) for i in result), dim=0)

def get_unique_states_with_exps(
        replay_path: str) -> typing.Tuple[
            torch.tensor, typing.List[replay_buffer.Experience]]:
    """Gets the unique states and a corresponding representative experience
    for each state."""
    result = dict()
    result_exps = []

    buffer = replay_buffer.FileReadableReplayBuffer(replay_path)
    try:
        for _ in range(len(buffer)):
            exp: replay_buffer.Experience = next(buffer)
            as_torch = torch.from_numpy(exp.encoded_state)
            if hash(as_torch) not in result_exps:
                result[hash(as_torch)] = [as_torch]
                result_exps.append(exp)
            elif all((existing != as_torch).sum() > 0 for existing in result[hash(as_torch)]):
                result[hash(as_torch)].append(as_torch)
    finally:
        buffer.close()

    cat_torch = torch.zeros((len(result_exps), list(result.values())[0].shape[0]))
    ctr = 0
    for arr in result.values():
        for res in arr:
            cat_torch[ctr, :] = res
            ctr += 1

    return cat_torch, result_exps

def main():
    """Main entry point for analyzing the model"""
    parser = argparse.ArgumentParser(description='Trains the deep.deep1 bot against a random bot')
    parser.add_argument('--py3', action='store_true', help='changes executable to python3')
    parser.add_argument('--numthreads', type=int, default=8,
                        help='number of threads to use for gathering experiences')
    parser.add_argument('--train_force_amount', type=float, default=0.1,
                        help='perc moves chosen by chance')
    parser.add_argument('--regul_factor', type=float, default=0.1,
                        help='regularization factor for criterion')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='unused; likelihood of surprising samples in training')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='unused importance sampling constant')
    parser.add_argument('--numexps', type=int, default=1000,
                        help='number of experiences to sample')
    parser.add_argument('--pca3d', action='store_true',
                        help='create the pca3d video')
    parser.add_argument('--mpf', type=float, default=16.67, help='milliseconds per frame')
    parser.add_argument('--marker_size', type=float, default=32, help='marker size for video')
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
                regul_factor=args.regul_factor, beta=args.beta, alpha=args.alpha,
                holdover=0, balance=True, balance_technique='action')
        ],
        cur_ind=0
    )
    deep_trainer._get_experiences_async( # pylint: disable=protected-access
        settings, executable, port, port+nthreads*10, 0,
        False, False, nthreads)

    network = deep2.Deep2Network.load(deep2.EVAL_MODELFILE)
    states, exps = get_unique_states_with_exps(deep2.REPLAY_FOLDER)
    network.eval()
    with torch.no_grad():
        labels = network(states)
    print(f'loaded {len(states)} states for analysis...')
    train_pwl = pwl.SimplePointWithLabelProducer(states, labels, 4, True)

    print('--fetching top 3 pcs--')
    traj: pca_gen.PCTrajectoryGen = pca_gen.find_trajectory(network, train_pwl, 3)
    if args.pca3d:
        print('--performing 3d plot--')

        if USE_MARKER_SHAPES:
            markers = MODULE + '._markers'
            ots = MODULE + '._ots'
            norm = MODULE + '._norm'
            cmap = 'cividis'
        else:
            print('--caching markers--')
            _cache_markers(_correctness_markers(network, states, exps))
            markers = MODULE + '._mark_cached_moves'
            ots = MODULE + '._ots_argmax'
            norm = MODULE + '._norm'
            cmap = 'Set1'

        print('--beginning plot--')
        pca_3d.plot_gen(traj, os.path.join(SAVEDIR, 'pca_3d'), True,
                        markers, ots, norm, cmap,
                        args.mpf, args.marker_size, None,
                        ['Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4',
                         'Layer 5', 'Layer 6', 'Output'])
    print('--plotting top 2 pcs--')
    pca_deep2.plot_trajectory(traj, os.path.join(SAVEDIR, 'pca'), exist_ok=True,
                              transparent=False, norm=mcolors.Normalize(-0.2, 0.2))
    print('--measuring participation ratio--')
    pr_traj: pr.PRTrajectory = pr.measure_pr_gen(network, train_pwl)
    print('--plotting participation ratio--')
    pr.plot_pr_trajectory(pr_traj, os.path.join(SAVEDIR, 'pr'), exist_ok=True)

    print('--finished--')

if __name__ == '__main__':
    main()
