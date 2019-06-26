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

import shared.filetools as filetools

import optimax_rogue.networking.serializer as ser
import or_reinforce.deep.replay_buffer as rb


class SessionSettings(ser.Serializable):
    """The settings for a single training session

    Attributes:
        tie_len (int): number of ticks before a tie is declared
        tar_ticks (int): the number of replay experiences that we are looking for
        train_force_amount (float): the percent of movements that are forced
    """
    def __init__(self, tie_len: int, tar_ticks: int, train_force_amount: float):
        self.tie_len = tie_len
        self.tar_ticks = tar_ticks
        self.train_force_amount = train_force_amount

    def __repr__(self):
        return (f'SessionSettings [tie_len={self.tie_len}, tar_ticks={self.tar_ticks}'
                + f', train_force_amount={self.train_force_amount}]')

ser.register(SessionSettings)

class TrainSettings(ser.Serializable):
    """The settings for training the deep1 bot

    Attributes:
        train_bot (str): the path to the bot to train module, including the callable
        adver_bot (str): the path to the adversary bot, including the callable
        bot_folder (str): the path to where the bot is storing model.pt and the replay folder
        train_seq (list[SessionSettings]): the list of training sessions to complete
        cur_ind (int): the index in train_seq that we are currently at.
    """
    def __init__(self, train_bot: str, adver_bot: str, bot_folder: str, train_seq: typing.List[SessionSettings], cur_ind: int):
        self.train_bot = train_bot
        self.adver_bot = adver_bot
        self.bot_folder = bot_folder
        self.train_seq = train_seq
        self.cur_ind = cur_ind

    @classmethod
    def defaults(cls):
        train_seq = [
            SessionSettings(tie_len=1000, tar_ticks=100000, train_force_amount=1), # need an extra one of these
        ]
        for i in range(20):
            train_seq.extend([
                SessionSettings(tie_len=1000, tar_ticks=100000, train_force_amount=1-(i*0.05)),
                SessionSettings(tie_len=5000, tar_ticks=100000, train_force_amount=1-(i*0.05))
            ])
        return cls(
            train_bot='or_reinforce.deep.deep1.deep1',
            adver_bot='optimax_rogue_bots.randombot.RandomBot',
            bot_folder=os.path.join('out', 'or_reinforce', 'deep', 'deep1'),
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
        return os.path.join(self.bot_folder, 'replay')

    @property
    def model_file(self):
        return os.path.join(self.bot_folder, 'model.pt')

    @property
    def bot_settings_file(self):
        return os.path.join(self.bot_folder, 'settings.json')

    @property
    def current_session(self) -> SessionSettings:
        return self.train_seq[self.cur_ind]

ser.register(TrainSettings)


SAVEDIR = shared.filetools.savepath()

def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description='Trains the deep.deep1 bot against a random bot')
    parser.add_argument('--port', type=int, default=1769, help='port to use')
    parser.add_argument('--headless', action='store_true', help='Use headless mode')
    parser.add_argument('--py3', action='store_true', help='changes executable to python3')
    parser.add_argument('--settings', type=str, default=os.path.join(SAVEDIR, 'settings.json'), help='path to the settings file')
    args = parser.parse_args()

    _run(args)

def _start_server(executable, secret1, secret2, port, max_ticks, create_flags):
    return subprocess.Popen(
        [executable, '-u', '-m', 'optimax_rogue.server.main', secret1, secret2, '--port', str(port),
         '--log', 'server_log.txt', '--tickrate', '0.01', '--dsunused', '--maxticks', str(max_ticks)],
        creationflags=create_flags)

def _start_bot(executable, bot, secret, port, create_flags, logfile):
    return subprocess.Popen(
        [executable, '-u', '-m', 'optimax_rogue_bots.main', 'localhost', str(port), bot, secret,
         '--log', logfile, '--tickrate', '0.01'],
        creationflags=create_flags
    )

def _start_spec(executable, port, create_flags):
    return subprocess.Popen(
        [executable, '-m', 'optimax_rogue_cmdspec.main', 'localhost', str(port)],
        creationflags=create_flags
    )

def _get_experiences(settings: TrainSettings, executable: str, port: int, create_flags: int, spec: bool):
    session: SessionSettings = settings.current_session
    num_ticks_to_do = session.tar_ticks
    if os.path.exists(settings.replay_folder):
        replay = rb.FileReadableReplayBuffer(settings.replay_folder)
        num_ticks_to_do -= len(replay)
        replay.close()

    with open(settings.bot_settings_file, 'w') as outfile:
        json.dump({'teacher_force_amt': session.train_force_amount}, outfile)

    while num_ticks_to_do > 0:
        print(f'--starting game to get another {num_ticks_to_do} experiences--')
        secret1 = secrets.token_hex()
        secret2 = secrets.token_hex()
        procs = []
        procs.append(_start_server(executable, secret1, secret2, port, session.tie_len, create_flags))
        if random.random() < 0.5:
            tmp = secret1
            secret1 = secret2
            secret2 = tmp
            del tmp

        time.sleep(2)

        procs.append(_start_bot(executable, settings.train_bot, secret1, port, create_flags, 'train_bot.log'))
        procs.append(_start_bot(executable, settings.adver_bot, secret2, port, create_flags, 'adver_bot.log'))
        if spec:
            procs.append(_start_spec(executable, port, create_flags))

        for proc in procs:
            proc.wait()

        print('--finished game--')
        time.sleep(0.5)
        replay = rb.FileReadableReplayBuffer(settings.replay_folder)
        num_ticks_to_do = session.tar_ticks - len(replay)
        replay.close()
        time.sleep(2)


def _train_experiences(settings: TrainSettings, executable: str):
    print('--training--')
    time.sleep(0.5)
    proc = subprocess.Popen(
        [executable, '-u', '-m', settings.train_bot]
    )
    proc.wait()
    print('--training finished--')
    time.sleep(0.5)

def _cleanup_session(settings: TrainSettings):
    filetools.deldir(settings.replay_folder)
    time.sleep(0.5)

def _run(args):
    settings: TrainSettings = None
    if not os.path.exists(args.settings):
        os.makedirs(os.path.dirname(args.settings))
        settings = TrainSettings.defaults()
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
    spec = not args.headless

    if os.path.exists(settings.replay_folder):
        rb.FileWritableReplayBuffer(settings.replay_folder, exist_ok=True).close()

    while settings.cur_ind < len(settings.train_seq):
        _get_experiences(settings, executable, port, create_flags, spec)
        _train_experiences(settings, executable)
        _cleanup_session(settings)
        settings.cur_ind += 1

if __name__ == '__main__':
    main()