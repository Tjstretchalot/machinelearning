"""Same as simplebot but now done with the QBot interface!"""
import os
import torch
import random

from shared.models.ff import FeedforwardComplex, FFTeacher
from shared.convutils import FluentShape

import or_reinforce.utils.qbot as qbot
import or_reinforce.utils.rewarders as rewarders
import or_reinforce.utils.encoders as encoders
import or_reinforce.utils.general as gen
from or_reinforce.utils.offline import OfflineLearner

from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
from optimax_rogue_bots.randombot import RandomBot


SAVEDIR = os.path.join('out', 'or_reinforce', 'simple', 'simplebot2')
MODELFILE = os.path.join(SAVEDIR, 'model.pt')

MOVE_MAP = [Move.Left, Move.Right, Move.Up, Move.Down]
VIEW_DISTANCE = 1

def _init_encoder(entity_iden):
    return encoders.MergedFlatEncoders(
        (
            encoders.MoveEncoder(MOVE_MAP),
            encoders.StaircaseEncoder(entity_iden),
            encoders.SurroundBarrierEncoder(entity_iden, VIEW_DISTANCE)
        )
    )

ENCODE_DIM = _init_encoder(None).dim

def _init_model():
    nets = FluentShape(ENCODE_DIM)
    return FeedforwardComplex(
        ENCODE_DIM, 1,
        [
            nets.linear_(50),
            nets.nonlin('isrlu'),
            nets.linear_(50),
            nets.nonlin('isrlu'),
            nets.linear_(50),
            nets.nonlin('isrlu'),
            nets.linear_(1),
        ])

class SimpleQBot(qbot.QBot):
    """A simple Q-bot

    Attributes:
        entity_iden (int): the entity we are controlling
        model (FeedforwardComplex): the model that does the evaluating
        teacher (FFTeacher): the teacher for the model
        optimizer (torch.nn.optimizer): the optimizer for the network
        criterion (callable): the evaluator for the network

        offline (OfflineLearner): the offline learner

        encoder (Encoder): the encoder
    """

    def __init__(self, entity_iden):
        self.entity_iden = entity_iden
        self.model = gen.init_or_load_model(_init_model, MODELFILE)
        self.teacher = FFTeacher()
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.003)
        self.criterion = torch.nn.MSELoss()
        self.encoder = _init_encoder(entity_iden)

        self.offline = OfflineLearner(self._learn, heap_size=10)

    @property
    def cutoff(self):
        return 3

    @property
    def alpha(self):
        return 0.3

    def evaluate(self, game_state: GameState, move: Move) -> float:
        result = torch.tensor([0.0], dtype=torch.float)
        self.teacher.classify(
            self.model,
            self.encoder.encode(game_state, move),
            result
            )
        return result

    def learn(self, game_state: GameState, move: Move, reward: float) -> None:
        self.offline(game_state, move, reward)

    def think(self, max_time: float):
        self.offline.think(max_time)

    def _learn(self, game_state: GameState, move: Move, reward: float) -> None:
        self.teacher.teach(
            self.model, self.optimizer, self.criterion,
            self.encoder.encode(game_state, move),
            torch.tensor([reward], dtype=torch.float32))
        return abs(reward)

    def save(self) -> None:
        gen.save_model(self.model, MODELFILE)


def simplebot2(entity_iden: int) -> 'Bot':
    """Creates a new simplebot2"""
    return qbot.QBotController(
        entity_iden,
        SimpleQBot(entity_iden),
        rewarders.SCRewarder(),
        MOVE_MAP,
        move_selstyle=qbot.QBotMoveSelectionStyle.Greedy,
        teacher=RandomBot(entity_iden, moves=MOVE_MAP),
        teacher_force_amt=0.2
    )
