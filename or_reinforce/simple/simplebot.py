"""Simple pathfinding model"""
import os
import torch
import sys
from collections import deque
import numpy as np
import random
import time

from optimax_rogue_bots.bot import Bot
from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
import optimax_rogue.networking.serializer as ser

from shared.models.ff import FeedforwardComplex, FFTeacher
from shared.convutils import FluentShape

SAVEDIR = os.path.join('out', 'or_reinforce', 'simple', 'simplebot')
MODELFILE = os.path.join(SAVEDIR, 'model.pt')

SIGHT_DIST = 1
MOVE_MAP = [Move.Left, Move.Right, Move.Up, Move.Down]
SC_POS_DIM = 2
ENCODE_DIM = ((SIGHT_DIST * 2 + 1) ** 2) + SC_POS_DIM + len(MOVE_MAP)
MOVE_LOOKUP = dict((val, key) for key, val in enumerate(MOVE_MAP))

def _encode(game_state: GameState, entity_iden: int, move: Move) -> torch.tensor:
    ent = game_state.iden_lookup[entity_iden]
    dung = game_state.world.dungeons[ent.depth]

    stairx, stairy = dung.staircase()
    deltax, deltay = stairx - ent.x, stairy - ent.y

    result = torch.zeros(ENCODE_DIM, dtype=torch.float32)
    result[0] = deltax
    result[1] = deltay
    result[MOVE_LOOKUP[move] + SC_POS_DIM] = 1

    realind = SC_POS_DIM + len(MOVE_MAP)
    for x in range(-SIGHT_DIST, SIGHT_DIST+1):
        realx = ent.x + x
        for y in range(-SIGHT_DIST, SIGHT_DIST+1):
            realy = ent.y + y
            if dung.is_blocked(realx, realy) or (ent.depth, realx, realy) in game_state.pos_lookup:
                result[realind] = 1

            realind += 1

    return result

def _decode(result: torch.tensor) -> float:
    return float(result[0])

def _reward(oldstate: GameState, newstate: GameState, entity_iden: int) -> float:
    ent_old = oldstate.iden_lookup[entity_iden]
    ent_new = newstate.iden_lookup[entity_iden]
    if ent_new.depth > ent_old.depth:
        return 1
    if (ent_new.x, ent_new.y) == (ent_old.x, ent_old.y):
        return -1

    staircase_old = oldstate.world.dungeons[ent_old.depth].staircase()
    staircase_new = newstate.world.dungeons[ent_new.depth].staircase()

    manh_old = abs(staircase_old[0] - ent_old.x) + abs(staircase_old[1] - ent_old.y)
    manh_new = abs(staircase_new[0] - ent_new.x) + abs(staircase_new[1] - ent_new.y)

    return 0.5 * (manh_old - manh_new)

def _init_model():
    nets = FluentShape(ENCODE_DIM)
    return FeedforwardComplex(
        ENCODE_DIM, 1,
        [
            nets.linear_(25),
            nets.nonlin('isrlu'),
            nets.linear_(25),
            nets.nonlin('isrlu'),
            nets.linear_(25),
            nets.nonlin('isrlu'),
            nets.linear_(1),
        ])

def _save_model(model: FeedforwardComplex):
    torch.save(model, MODELFILE)

def _load_model():
    return torch.load(MODELFILE)

def _init_or_load_model():
    os.makedirs(SAVEDIR, exist_ok=True)

    if os.path.exists(MODELFILE):
        return _load_model()
    model = _init_model()
    _save_model(model)
    return model

ALPHA = 0.3
CUTOFF = 3

class SimpleBot(Bot):
    """Simple pathfinding bot

    Attributes:
        history (deque[GameState]): recent game states, where the left corresponds to len(history)
            ticks ago and the right corresponds to the last tick

        model (FeedforwardComplex): the model that predicts q-values
        teacher (FFTeacher): the teacher for the model
        optimizer (torch.nn.Optimizer): the optimizer
        criterion (callable): criterion
    """

    def __init__(self, entity_iden: int):
        super().__init__(entity_iden)
        self.model = _init_or_load_model()
        self.history = deque()
        self.teacher = FFTeacher()
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.003)
        self.criterion = torch.nn.MSELoss()

        self.spam_loss = False
        self.spam_moves = False
        self.print_loss_improves = True
        self.random_perc = 0.2
        self.best_loss = float('inf')
        self.next_save = 50

    def move(self, game_state: GameState):
        gs_copy = ser.deserialize(ser.serialize(game_state))
        self.history.append((gs_copy, None))

        if len(self.history) == CUTOFF + 1:
            self.teach()

        move = self.eval(game_state)
        if np.random.uniform(0, 1) < self.random_perc:
            move = random.choice(MOVE_MAP)
        self.history.pop()
        self.history.append((gs_copy, move))

        self.next_save -= 1
        if self.next_save <= 0:
            self.save()
            self.next_save = 50

        return move

    def finished(self, game_state: GameState, result):
        self.save()

    def save(self):
        """saves the model"""
        print(f'[simplebot] {time.ctime()} saving')
        sys.stdout.flush()
        _save_model(self.model)

    def teach(self):
        """Must be called when we have CUTOFF+1 history. Takes the oldest history item, calculates
        the value for the finite series of diminished rewards, and then trains the network
        on that"""
        original, og_move = self.history.popleft()
        previous = original
        penalty = 1
        reward = 0
        for i in range(CUTOFF):
            reward += penalty * _reward(previous, self.history[i][0], self.entity_iden)
            previous = self.history[i][0]
            penalty *= ALPHA


        loss = self.teacher.teach(
            self.model, self.optimizer, self.criterion,
            _encode(original, self.entity_iden, og_move),
            torch.tensor([reward], dtype=torch.float32))
        if self.spam_loss:
            print(f'[simplebot] loss={loss}')
            sys.stdout.flush()
        if self.print_loss_improves:
            if loss < self.best_loss:
                self.best_loss = loss
                print(f'[simplebot] loss improved to {loss} for move '
                      + f'{og_move.name} reward {reward}')
                sys.stdout.flush()

    def eval(self, game_state: GameState) -> Move:
        """Chooses the best move according to our model for the given state"""
        scores = []
        out = torch.tensor([0.0])
        for move in MOVE_MAP:
            self.teacher.classify(
                self.model,
                _encode(game_state, self.entity_iden, move),
                out)
            scores.append(out.item())
        if self.spam_moves:
            toprint = []
            for ind, move in enumerate(MOVE_MAP):
                toprint.extend((str(move), ': ', f'{scores[ind]:.3f}'))
            print('{' + ', '.join(toprint) + '}')
            sys.stdout.flush()
        return MOVE_MAP[int(np.argmax(scores))]
