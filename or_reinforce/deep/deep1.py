"""A temporal difference learner with a multi-layer network with batch normalization as a nonlinear
function approximator, which learns offline through a large replay buffer."""

import os
import torch
import typing
import json
import math

from shared.teachers import NetworkTeacher, Network
from shared.models.ff import FeedforwardComplex, FFTeacher
from shared.convutils import FluentShape
import shared.trainer as tnr
import shared.perf_stats as perf_stats
import shared.criterion as crits

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
ALPHA = 0.5
CUTOFF = 10
PRED_WEIGHT = ALPHA ** CUTOFF

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
            nets.nonlin('tanh')
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
        evaluation (bool): True to not store experiences, False to store experiences

        write_replay_buffer (WritableReplayBuffer, optional): the buffer for replays

        encoder (Encoder): the encoder
    """
    def __init__(self, entity_iden: int, replay_path=REPLAY_FOLDER, evaluation=False):
        self.entity_iden = entity_iden
        self.model = gen.init_or_load_model(_init_model, MODELFILE)
        self.teacher = FFTeacher()
        self.evaluation = evaluation
        self.encoder = _init_encoder(entity_iden)

        if not evaluation:
            self.replay = replay_buffer.FileWritableReplayBuffer(replay_path, exist_ok=True)
        else:
            self.replay = None

    def __call__(self, entity_iden):
        self.entity_iden = entity_iden
        self.encoder = _init_encoder(entity_iden)

    @property
    def cutoff(self):
        return CUTOFF

    @property
    def alpha(self):
        return ALPHA

    def evaluate(self, game_state: GameState, move: Move):
        result = torch.tensor([0.0], dtype=torch.float)
        self.teacher.classify(
            self.model,
            self.encoder.encode(game_state, move),
            result
            )
        return float(result.item())

    def learn(self, game_state: GameState, move: Move, new_state: GameState,
              reward_raw: float, reward_pred: float) -> None:
        if self.evaluation:
            print(f'predicted reward: {self.evaluate(game_state, move):.2f} vs actual reward '
                  + f'{reward_raw:.2f} + {reward_pred:.2f} = {reward_raw + reward_pred:.2f}')
            return
        player_id = 1 if self.entity_iden == game_state.player_1_iden else 2
        self.replay.add(replay_buffer.Experience(game_state, move, self.cutoff,
                                                 new_state, reward_raw, player_id))

    def save(self) -> None:
        pass

class MyQBotController(qbot.QBotController):
    """Adds the pitch and supported moves for this to the qbot controller"""
    @classmethod
    def pitch(cls):
        return (
            'DeepQBot',
            'Learns through offline replay of moves with a target network',
        )

    @classmethod
    def supported_moves(cls):
        return MOVE_MAP

def deep1(entity_iden: int, settings: str = None) -> 'Bot':
    """Creates a new simplebot2"""
    if not settings:
        settings = SETTINGS_FILE
    if not os.path.exists(settings):
        raise FileNotFoundError(settings)

    with open(settings, 'r') as infile:
        settings = json.load(infile)

    return MyQBotController(
        entity_iden,
        DeepQBot(entity_iden, settings['replay_path'], (('eval' in settings) and settings['eval'])),
        rewarders.SCRewarder(bigreward=(1 - ALPHA)), # biggest cumulative reward is 1
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
        target_model (Network): the network we use to evaluate states
        target_teacher (NetworkTeacher): the teacher we can use to evaluate states
        encoders_by_id (dict[int, Encoder]): the encoders by player id

        _buffer [torch.tensor[n x ENCODE_DIM]]: where n is len(MOVE_MAP) times the largest batch
            size we've seen so far. Used to calculate the predicted reward using our target network

        _outbuffer [torch.tensor[n]]: used for storing the rewards
    """
    def __init__(self, replay: replay_buffer.ReadableReplayBuffer, target_model: Network,
                 target_teacher: NetworkTeacher):
        super().__init__(len(replay), ENCODE_DIM, 1)
        self.replay = replay
        self.target_model = target_model
        self.target_teacher = target_teacher
        self.encoders_by_id = dict()

        self._buffer = torch.zeros(len(MOVE_MAP), ENCODE_DIM, dtype=torch.float32)
        self._outbuffer = torch.zeros(len(MOVE_MAP), dtype=torch.float32)

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

        for i, move in enumerate(MOVE_MAP):
            enc.encode(exp.new_state, move, out=self._buffer[i])

        self.target_teacher.classify_many(self.target_model, self._buffer[:len(MOVE_MAP)],
                                          self._outbuffer[:len(MOVE_MAP)])

        return pwl.PointWithLabel(
            point=point_tens,
            label=exp.reward_rec + float(self._outbuffer[:len(MOVE_MAP)].max().item()) * PRED_WEIGHT
        )

    def _ensure_buffers(self, batch_size):
        req_size = batch_size * len(MOVE_MAP)
        if self._outbuffer.shape[0] >= req_size:
            return

        self._buffer = torch.zeros((req_size, ENCODE_DIM), dtype=torch.float32)
        self._outbuffer = torch.zeros(req_size, dtype=torch.float32)

    def fill(self, points: torch.tensor, labels: torch.tensor) -> None:
        batch_size = points.shape[0]
        self._ensure_buffers(batch_size)
        lmmap = len(MOVE_MAP)

        exps = self.replay.sample(batch_size)
        for i, exp in enumerate(exps):
            ent_id = exp.state.player_1_iden if exp.player_id == 1 else exp.state.player_2_iden
            if ent_id not in self.encoders_by_id:
                self.encoders_by_id[ent_id] = _init_encoder(ent_id)

            enc: encoders.FlatEncoder = self.encoders_by_id[ent_id]
            enc.encode(exp.state, exp.action, out=points[i])

            for j, move in enumerate(MOVE_MAP):
                enc.encode(exp.new_state, move, out=self._buffer[i * lmmap + j])
            labels[i] = exp.reward_rec

        self.target_teacher.classify_many(self.target_model, self._buffer[:batch_size * lmmap],
                                          self._outbuffer[:batch_size * lmmap].unsqueeze(1))

        best_preds, _ = self._outbuffer[:batch_size * lmmap].reshape(batch_size, lmmap).max(dim=1)
        labels += best_preds * PRED_WEIGHT



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
    perf_file = os.path.join(SAVEDIR, 'offline_learning_perf.log')
    perf = perf_stats.LoggingPerfStats('deep1 offline learning', perf_file)

    replay = replay_buffer.FileReadableReplayBuffer(REPLAY_FOLDER, perf=perf)

    print(f'loaded {len(replay)} experiences for replay...')

    network = gen.init_or_load_model(_init_model, MODELFILE)
    teacher = MyTeacher(FFTeacher())


    train_pwl = MyPWL(replay, gen.load_model(MODELFILE), teacher)
    test_pwl = train_pwl

    def update_target(ctx: tnr.GenericTrainingContext, hint: str):
        ctx.logger.info('swapping target network, hint=%s', hint)
        gen.save_model(network, MODELFILE)
        train_pwl.target_model = gen.load_model(MODELFILE)

    trainer = tnr.GenericTrainer(
        train_pwl=train_pwl,
        test_pwl=test_pwl,
        teacher=teacher,
        batch_size=32,
        learning_rate=0.001,
        optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.001),
        criterion=torch.nn.MSELoss()
    )
    (trainer
     .reg(tnr.EpochsTracker())
     .reg(tnr.EpochsStopper(100))
     .reg(tnr.InfOrNANDetecter())
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayStopper(1))
     .reg(tnr.OnEpochCaller.create_every(update_target, start=2, skip=2))
     .reg(tnr.DecayOnPlateau())
    )
    trainer.train(network, target_dtype=torch.float32, point_dtype=torch.float32, perf=perf)

    gen.save_model(network, MODELFILE)

    replay.close()

if __name__ == '__main__':
    offline_learning()
