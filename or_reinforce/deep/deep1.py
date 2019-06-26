"""A temporal difference learner with a multi-layer network with batch normalization as a nonlinear
function approximator, which learns offline through a large replay buffer."""

import os
import torch
import typing
import json

from shared.teachers import NetworkTeacher, Network
from shared.models.ff import FeedforwardComplex, FFTeacher
from shared.convutils import FluentShape
import shared.trainer as tnr

import or_reinforce.utils.qbot as qbot
import or_reinforce.utils.rewarders as rewarders
import or_reinforce.utils.encoders as encoders
import or_reinforce.utils.general as gen
import or_reinforce.deep.replay_buffer as replay_buffer

from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
from optimax_rogue_bots.randombot import RandomBot

import shared.pwl as pwl


SAVEDIR = os.path.join('out', 'or_reinforce', 'simple', 'simplebot3')
MODELFILE = os.path.join(SAVEDIR, 'model.pt')

MOVE_MAP = [Move.Left, Move.Right, Move.Up, Move.Down]

def _init_encoder(entity_iden):
    return encoders.MergedFlatEncoders(
        (
            encoders.MoveEncoder(MOVE_MAP),
            encoders.LocationEncoder(entity_iden),
            encoders.StaircaseLocationEncoder(entity_iden),
        )
    )

ENCODE_DIM = _init_encoder(None).dim

def _init_model():
    nets = FluentShape(ENCODE_DIM)
    return FeedforwardComplex(
        ENCODE_DIM, 1,
        [
            nets.batch_norm(),
            nets.linear_(50),
            nets.nonlin('tanh'),
            nets.batch_norm(),
            nets.linear_(50),
            nets.nonlin('tanh'),
            nets.batch_norm(),
            nets.linear_(50),
            nets.nonlin('tanh'),
            nets.batch_norm(),
            nets.linear_(1),
        ])

SAVEDIR = os.path.join('out', 'or_reinforce', 'deep', 'deep1')
MODELFILE = os.path.join(SAVEDIR, 'model.pt')
REPLAY_FOLDER = os.path.join(SAVEDIR, 'replay')
SETTINGS_FILE = os.path.join(SAVEDIR, 'settings.json')

class DeepQBot(qbot.QBot):
    """The Q-bot implementation

    Attributes:
        entity_iden (int): the entity we are controlling
        model (FeedforwardComplex): the model that does the evaluating
        teacher (FFTeacher): the teacher for the model

        write_replay_buffer (WritableReplayBuffer, optional): the buffer for replays

        encoder (Encoder): the encoder
    """
    def __init__(self, entity_iden: int):
        self.entity_iden = entity_iden
        self.model = gen.init_or_load_model(_init_model, MODELFILE)
        self.teacher = FFTeacher()
        self.encoder = _init_encoder(entity_iden)

        self.replay = replay_buffer.FileWritableReplayBuffer(REPLAY_FOLDER, exist_ok=True)

    def __call__(self, entity_iden):
        self.entity_iden = entity_iden
        self.encoder = _init_encoder(entity_iden)

    @property
    def cutoff(self):
        return 1

    @property
    def alpha(self):
        return 0.9

    def evaluate(self, game_state: GameState, move: Move):
        result = torch.tensor([0.0], dtype=torch.float)
        self.teacher.classify(
            self.model,
            self.encoder.encode(game_state, move),
            result
            )
        return float(result.item())

    def learn(self, game_state: GameState, move: Move, reward: float) -> None:
        player_id = 1 if self.entity_iden == game_state.player_1_iden else 2
        self.replay.add(replay_buffer.Experience(game_state, move, reward, player_id))

        if len(self.replay) % 1000 == 0:
            print(f'[DeepQBot] replay size = {len(self.replay)}')

    def save(self) -> None:
        pass

def deep1(entity_iden: int) -> 'Bot':
    """Creates a new simplebot2"""
    if not os.path.exists(SETTINGS_FILE):
        raise FileNotFoundError(SETTINGS_FILE)

    with open(SETTINGS_FILE, 'r') as infile:
        settings = json.load(infile)

    return qbot.QBotController(
        entity_iden,
        DeepQBot(entity_iden),
        rewarders.SCRewarder(),
        MOVE_MAP,
        move_selstyle=qbot.QBotMoveSelectionStyle.Greedy,
        teacher=RandomBot(entity_iden, moves=MOVE_MAP),
        teacher_force_amt=settings['teacher_force_amt']
    )

class MyPWL(pwl.PointWithLabelProducer):
    """The point with label producer that just encodes the experiences and replays
    in the standard points with labels format

    Attributes:
        replay (ReplayBuffer): the buffer we are loading experiences from
        encoders_by_id (dict[int, Encoder]): the encoders by player id
    """
    def __init__(self, replay: replay_buffer.ReadableReplayBuffer):
        super().__init__(len(replay), ENCODE_DIM, 1)
        self.replay = replay
        self.encoders_by_id = dict()

    def mark(self):
        self.replay.mark()

    def reset(self):
        self.replay.reset()

    @property
    def position(self):
        return self.replay.position

    @position.setter
    def position(self, val):
        raise NotImplementedError

    @property
    def remaining_in_epoch(self):
        return self.replay.remaining_in_epoch

    def __next__(self) -> pwl.PointWithLabel:
        exp: replay_buffer.Experience = next(self.replay)

        ent_id = exp.state.player_1_iden if exp.player_id == 1 else exp.state.player_2_iden
        if ent_id not in self.encoders_by_id:
            self.encoders_by_id[ent_id] = _init_encoder(ent_id)

        enc: encoders.FlatEncoder = self.encoders_by_id[ent_id]
        point_tens = enc.encode(exp.game_state, exp.action)
        return pwl.PointWithLabel(point=point_tens, label=exp.reward)

    def fill(self, points: torch.tensor, labels: torch.tensor) -> None:
        batch_size = points.shape[0]
        exps = self.replay.sample(batch_size)
        for i, exp in enumerate(exps):
            ent_id = exp.state.player_1_iden if exp.player_id == 1 else exp.state.player_2_iden
            if ent_id not in self.encoders_by_id:
                self.encoders_by_id[ent_id] = _init_encoder(ent_id)

            enc: encoders.FlatEncoder = self.encoders_by_id[ent_id]
            enc.encode(exp.state, exp.action, out=points[i])
            labels[i] = exp.reward

    def _fill(self, points, labels):
        raise NotImplementedError

    def _position(self, pos):
        raise NotImplementedError

class MyTeacher(NetworkTeacher):
    """A teacher that maps from labels from just batch_size to batch_sizex1 by adding
    a dummy dimension

    Attributes:
        teacher (FFTeacher): the real teacher
    """

    def __init__(self, teacher: FFTeacher):
        self.teacher = teacher

    def teach_many(self, network: Network, optimizer: torch.optim.Optimizer, criterion: typing.Any,
                   points: torch.tensor, labels: torch.tensor) -> float:
        return self.teacher.teach_many(network, optimizer, criterion, points, labels.unsqueeze(1))

    def classify_many(self, network: Network, points: torch.tensor, out: torch.tensor):
        return self.teacher.classify_many(network, points, out)


def offline_learning():
    """Loads the replay buffer and trains on it."""
    replay = replay_buffer.FileReadableReplayBuffer(REPLAY_FOLDER)

    print(f'loaded {len(replay)} experiences for replay...')

    train_pwl = MyPWL(replay)
    test_pwl = train_pwl

    network = gen.init_or_load_model(_init_model, MODELFILE)

    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=MyTeacher(FFTeacher()),
        batch_size=32,
        learning_rate=0.003,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.003),
        criterion=torch.nn.SmoothL1Loss()
    )
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(100))
     .reg(tnr.InfOrNANDetecter())
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(8))
     .reg(tnr.LRMultiplicativeDecayer())
     .reg(tnr.DecayOnPlateau())
    )
    trainer.train(network, target_dtype=torch.float32, point_dtype=torch.float32)

    gen.save_model(network, MODELFILE)

    replay.close()

if __name__ == '__main__':
    offline_learning()
