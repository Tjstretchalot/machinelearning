"""Launches a game with two bots and watches them with a command spectator
"""
import shared.setup_torch # pylint: disable=unused-import
import argparse
import subprocess
import secrets
import time
import typing
import os
import json
import random
import psutil
import math
import sys
import numpy as np

from multiprocessing import Process

import shared.filetools as filetools

import optimax_rogue.logic.moves as moves
import optimax_rogue.networking.serializer as ser
import or_reinforce.deep.replay_buffer as rb

MOVES = list(moves.Move)
MOVES.remove(moves.Move.Stay)

BALANCE_LOOKUP = {
    'reward': {
        'exp_types': [rb.PositiveExperience(), rb.NegativeExperience()],
        'style': 'desc'
    },
    'action': {
        'exp_types': [rb.ActionExperience(mv) for mv in MOVES],
        'style': 'exact'
    }
}

class SessionSettings(ser.Serializable):
    """The settings for a single training session

    Attributes:
        tie_len (int): number of ticks before a tie is declared
        tar_ticks (int): the number of replay experiences that we are looking for
        train_force_amount (float): the percent of movements that are forced
        regul_factor (float): the regularization factor for training

        holdover (int): the number of experiences from this session (and previous)
            that should be held over to the next session

        balance (bool): True if the dataset is balanced somehow, False otherwise
        balance_technique (str, optional): one of 'reward' or 'action'. Reward
            balances the positive/negative experiences, action balances the move
            selected

    """
    def __init__(self, tie_len: int, tar_ticks: int, train_force_amount: float,
                 regul_factor: float, holdover: int,
                 balance: bool, balance_technique: typing.Optional[str] = None):
        self.tie_len = tie_len
        self.tar_ticks = tar_ticks
        self.train_force_amount = train_force_amount
        self.regul_factor = regul_factor
        self.holdover = holdover
        self.balance = balance
        self.balance_technique = balance_technique

    @classmethod
    def identifier(cls):
        return 'or_reinforce.runners.deep_trainer.session_settings'

    def __repr__(self):
        return (f'SessionSettings [tie_len={self.tie_len}, tar_ticks={self.tar_ticks}\n'
                + f', train_force_amount={self.train_force_amount}\n'
                + f', regul_factor={self.regul_factor}, balance={self.balance}\n'
                + f', balance_technique={self.balance_technique}\n'
                + f', holdover={self.holdover}]')

ser.register(SessionSettings)

class TrainSettings(ser.Serializable):
    """The settings for training the deep bot

    Attributes:
        train_bot (str): the path to the bot to train module, including the callable
        adver_bot (str): the path to the adversary bot, including the callable
        bot_folder (str): the path to where the bot is storing model.pt and the replay folder
        train_seq (list[SessionSettings]): the list of training sessions to complete
        cur_ind (int): the index in train_seq that we are currently at.
    """
    def __init__(self, train_bot: str, adver_bot: str, bot_folder: str,
                 train_seq: typing.List[SessionSettings], cur_ind: int):
        self.train_bot = train_bot
        self.adver_bot = adver_bot
        self.bot_folder = bot_folder
        self.train_seq = train_seq
        self.cur_ind = cur_ind

    @classmethod
    def identifier(cls):
        return 'or_reinforce.runners.deep_trainer.train_settings'

    @classmethod
    def defaults(cls, train_bot: str):
        """Gets the current recommended settings for training the deep bot as well as possible.
        This can take some time."""
        train_seq = []
        # first session short to avoid norms being way off
        train_seq.append(SessionSettings(tie_len=111, tar_ticks=1000, train_force_amount=1,
                                         regul_factor=6, holdover=10000,
                                         balance=True, balance_technique='action'))
        for i in range(5): # 5 * 2k = 10k samples random
            train_seq.append(SessionSettings(tie_len=111, tar_ticks=2000, train_force_amount=1,
                                             regul_factor=5 - i, holdover=10000,
                                             balance=True, balance_technique='action'))

        for tfa in np.linspace(1, 0.1, 25): # 25*4k = 100k samples linearly decreasing tfa
            train_seq.extend([
                SessionSettings(tie_len=111, tar_ticks=2000, train_force_amount=float(tfa),
                                regul_factor=tfa, holdover=10000, balance=True,
                                balance_technique='action'),
                SessionSettings(tie_len=111, tar_ticks=2000, train_force_amount=float(tfa),
                                regul_factor=tfa, holdover=10000, balance=True,
                                balance_technique='action')
            ])
        return cls(
            train_bot=train_bot,
            adver_bot='optimax_rogue_bots.randombot.RandomBot',
            bot_folder=os.path.join('out', *train_bot.split('.')[:-1]),
            train_seq=train_seq,
            cur_ind=0
        )

    def to_prims(self):
        return {
            'train_bot': self.train_bot,
            'adver_bot': self.adver_bot,
            'bot_folder': self.bot_folder,
            'train_seq': list(ser.serialize_embeddable(ses) for ses in self.train_seq),
            'cur_ind': self.cur_ind
        }

    @classmethod
    def from_prims(cls, prims):
        return cls(
            train_bot=prims['train_bot'],
            adver_bot=prims['adver_bot'],
            bot_folder=prims['bot_folder'],
            train_seq=list(ser.deserialize_embeddable(ses) for ses in prims['train_seq']),
            cur_ind=prims['cur_ind']
        )

    @property
    def replay_folder(self):
        """Gets the folder which replays are stored by default for the bot"""
        return os.path.join(self.bot_folder, 'replay')

    @property
    def model_file(self):
        """Gets the file where the bots model is stored"""
        return os.path.join(self.bot_folder, 'model.pt')

    @property
    def bot_settings_file(self):
        """Gets the default file for the bots settings"""
        return os.path.join(self.bot_folder, 'settings.json')

    @property
    def current_session(self) -> SessionSettings:
        """Gets the current session settings"""
        return self.train_seq[self.cur_ind]

    @property
    def bot_module(self) -> str:
        """Get the module that the bot resides in, i.e., optimax_rogue_bots.randombot"""
        return '.'.join(self.train_bot.split('.')[:-1])

ser.register(TrainSettings)


SAVEDIR = shared.filetools.savepath()
HOLDOVER_DIR = os.path.join(SAVEDIR, 'holdover_replay')

def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description='Trains the deep bot against a random bot')
    parser.add_argument('--port', type=int, default=1769, help='port to use')
    parser.add_argument('--headless', action='store_true', help='Use headless mode')
    parser.add_argument('--py3', action='store_true', help='changes executable to python3')
    parser.add_argument('--settings', type=str, default=os.path.join(SAVEDIR, 'settings.json'),
                        help='path to the settings file')
    parser.add_argument('--aggressive', action='store_true',
                        help='no sleeps, use as much cpu as possible')
    parser.add_argument('--numthreads', type=int, default=10,
                        help='number of threads to use for gathering experiences')
    parser.add_argument('--debug', action='store_true',
                        help='stop after first training session completes')
    parser.add_argument('bot', type=str, help='the bot to train, including the callable')
    args = parser.parse_args()

    _run(args)

def _start_server(executable, secret1, secret2, port, max_ticks, aggressive, create_flags):
    args = [
        executable, '-u', '-m', 'optimax_rogue.server.main', secret1, secret2, '--port', str(port),
        '--log', 'server_log.txt', '--dsunused', '--maxticks', str(max_ticks), '--tickrate'
    ]
    if aggressive:
        args.append('0')
        args.append('--aggressive')
    else:
        args.append('0.01')

    return subprocess.Popen(
        args,
        creationflags=create_flags)

def _start_bot(executable, bot, secret, port, create_flags, aggressive, logfile, add_args=None):
    args = [
        executable, '-u', '-m', 'optimax_rogue_bots.main', 'localhost', str(port), bot, secret,
        '--log', logfile, '--tickrate'
    ]
    if aggressive:
        args.append('0')
        args.append('--aggressive')
    else:
        args.append('0.01')

    if add_args:
        args.extend(add_args)

    return subprocess.Popen(
        args,
        creationflags=create_flags
    )

def _start_spec(executable, port, create_flags):
    return subprocess.Popen(
        [executable, '-m', 'optimax_rogue_cmdspec.main', 'localhost', str(port)],
        creationflags=create_flags
    )

def _get_experiences_sync(settings: TrainSettings, executable: str, port_chooser: typing.Callable,
                          create_flags: int, aggressive: bool, spec: bool,
                          replaypath: str, settings_path: str, tar_num_ticks: int):
    session: SessionSettings = settings.current_session
    num_ticks_to_do = tar_num_ticks
    if os.path.exists(replaypath):
        replay = rb.FileReadableReplayBuffer(replaypath)
        num_ticks_to_do -= len(replay)
        replay.close()

    with open(settings_path, 'w') as outfile:
        json.dump({'teacher_force_amt': session.train_force_amount,
                   'replay_path': replaypath}, outfile)

    while num_ticks_to_do > 0:
        print(f'--starting game to get another {num_ticks_to_do} experiences--')
        sys.stdout.flush()
        secret1 = secrets.token_hex()
        secret2 = secrets.token_hex()
        port = port_chooser()
        procs = []
        procs.append(_start_server(executable, secret1, secret2, port, session.tie_len,
                                   aggressive, create_flags))
        if random.random() < 0.5:
            tmp = secret1
            secret1 = secret2
            secret2 = tmp
            del tmp

        time.sleep(2)

        procs.append(_start_bot(executable, settings.train_bot, secret1, port, create_flags,
                                aggressive, 'train_bot.log', ['--settings', settings_path]))
        procs.append(_start_bot(executable, settings.adver_bot, secret2, port, create_flags,
                                aggressive, 'adver_bot.log'))
        if spec:
            procs.append(_start_spec(executable, port, create_flags))

        for proc in procs:
            proc.wait()

        print('--finished game--')
        sys.stdout.flush()
        time.sleep(0.5)
        if not os.path.exists(replaypath):
            print('--game failed unexpectedly (no replay), waiting a bit and restarting--')
            sys.stdout.flush()
        else:
            replay = rb.FileReadableReplayBuffer(replaypath)
            num_ticks_to_do = tar_num_ticks - len(replay)
            replay.close()
            if num_ticks_to_do <= 0 and session.balance:
                rb.balance_experiences(
                    replaypath, **BALANCE_LOOKUP[session.balance_technique])
                replay = rb.FileReadableReplayBuffer(replaypath)
                num_ticks_to_do = tar_num_ticks - len(replay)
                replay.close()
        time.sleep(2)

class PortChooser:
    """Selects a port between the specified minimum and the minimum plus the specified amount

    Attributes:
        port_min (int): the minimum port to select
        nports (int): the number of ports this has available to it
        offset (int): the current offset from port_min for the next result
    """

    def __init__(self, port_min: int, nports: int):
        self.port_min = port_min
        self.nports = nports
        self.offset = 0

    def __call__(self):
        res = self.port_min + self.offset
        self.offset = (self.offset + 1) % self.nports
        return res

def _get_experiences_target(serd_settings: dict, executable: str, port_min: int, port_max: int,
                            create_flags: int, aggressive: bool, spec: bool, replay_path: str,
                            settings_path: str, tar_num_ticks: int):
    settings = ser.deserialize_embeddable(serd_settings)
    _get_experiences_sync(settings, executable, PortChooser(port_min, port_max-port_min),
                          create_flags, aggressive, spec, replay_path, settings_path, tar_num_ticks)

def _get_experiences_async(settings: TrainSettings, executable: str, port_min: int, port_max: int,
                           create_flags: int, aggressive: bool, spec: bool, nthreads: int):
    num_ticks_to_do = settings.current_session.tar_ticks
    if os.path.exists(settings.replay_folder):
        replay = rb.FileReadableReplayBuffer(settings.replay_folder)
        num_ticks_to_do -= len(replay)
        replay.close()

        if num_ticks_to_do <= 0:
            print(f'get_experiences_async nothing to do (already at {settings.replay_folder}')
            return

    replay_paths = [os.path.join(settings.bot_folder, f'replay_{i}') for i in range(nthreads)]
    setting_paths = [os.path.join(settings.bot_folder, f'settings_{i}.json') for i in range(nthreads)]
    workers = []
    serd_settings = ser.serialize_embeddable(settings)
    ports_per = (port_max - port_min) // nthreads
    if ports_per < 3:
        raise ValueError(f'not enough ports assigned ({nthreads} threads, {port_max-port_min} ports)')
    ticks_per = int(math.ceil(num_ticks_to_do / nthreads))
    for worker in range(nthreads):
        proc = Process(target=_get_experiences_target,
                       args=(serd_settings, executable, port_min + worker*ports_per,
                             port_min + (worker+1)*ports_per, create_flags, aggressive, spec,
                             replay_paths[worker], setting_paths[worker], ticks_per))
        proc.start()
        workers.append(proc)
        time.sleep(1)

    for proc in workers:
        proc.join()

    print(f'get_experiences_async finished, storing in {settings.replay_folder}')
    if os.path.exists(settings.replay_folder):
        filetools.deldir(settings.replay_folder)

    if os.path.exists(settings.replay_folder):
        tmp_replay_folder = settings.replay_folder + '_tmp'
        os.rename(settings.replay_folder, tmp_replay_folder)
        replay_paths.append(tmp_replay_folder)

    if os.path.exists(HOLDOVER_DIR):
        replay_paths.append(HOLDOVER_DIR)

    rb.merge_buffers(replay_paths, settings.replay_folder)

    for path in replay_paths:
        filetools.deldir(path)


def _train_experiences(settings: TrainSettings, executable: str):
    print('--training--')
    stime = time.time()
    time.sleep(0.5)
    proc = subprocess.Popen(
        [executable, '-u', '-m', settings.bot_module, str(settings.current_session.regul_factor)]
    )
    proc.wait()
    print('--training finished--')
    time.sleep(0.5)
    if 60 + stime > time.time():
        time.sleep((60 + stime) - time.time())

def _cleanup_session(settings: TrainSettings):
    if settings.current_session.holdover <= 0:
        filetools.deldir(settings.replay_folder)
        time.sleep(0.5)
        return

    os.rename(settings.replay_folder, HOLDOVER_DIR)
    rb.ensure_max_length(HOLDOVER_DIR, settings.current_session.holdover)

def _run(args):
    settings: TrainSettings = None
    if not os.path.exists(args.settings):
        os.makedirs(os.path.dirname(args.settings))
        settings = TrainSettings.defaults(args.bot)
        with open(args.settings, 'w') as outfile:
            json.dump(ser.serialize_embeddable(settings), outfile)
    else:
        with open(args.settings, 'r') as infile:
            rawsettings = json.load(infile)
            settings = ser.deserialize_embeddable(rawsettings)

    os.makedirs(settings.bot_folder, exist_ok=True)
    executable = 'python3' if args.py3 else 'python'
    port = args.port
    create_flags = 0 if args.headless else subprocess.CREATE_NEW_CONSOLE
    nthreads = args.numthreads
    spec = not args.headless

    ncores = psutil.cpu_count(logical=False)
    if args.aggressive and nthreads > (ncores // 3):
        print(f'auto-reducing simultaneous servers to {ncores // 3} since there are {ncores} '
              + 'cores and we need 3 cores per server')
        nthreads = ncores // 3

    if os.path.exists(settings.replay_folder):
        rb.FileWritableReplayBuffer(settings.replay_folder, exist_ok=True).close()

    while settings.cur_ind < len(settings.train_seq):
        print('--starting session--')
        print(settings.current_session)
        _get_experiences_async(settings, executable, port, port + 10*nthreads, create_flags,
                               args.aggressive, spec, nthreads)
        _train_experiences(settings, executable)
        if args.debug:
            break
        _cleanup_session(settings)
        settings.cur_ind += 1
        with open(args.settings, 'w') as outfile:
            json.dump(ser.serialize_embeddable(settings), outfile)

    print('--finished--')

if __name__ == '__main__':
    main()
