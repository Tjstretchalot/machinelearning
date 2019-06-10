"""Describes a bot that learns through some variation of Q-learning. Key
characteristics of such bots are:
    1. They have some (explicit) reward function
    2. They can learn from moves they didn't take
"""
from collections import deque
import random
import typing
import enum
import numpy as np
import torch
import time

from optimax_rogue_bots.bot import Bot
from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
from optimax_rogue.logic.updater import UpdateResult

from or_reinforce.utils.gstate_cache import GameStateCache

class Rewarder:
    """The interface for something that rewards the qbot"""

    def reward(self, entity_iden: int, oldstate: GameState, newstate: GameState) -> float:
        """Evaluates the reward for going from the first to second state"""
        raise NotImplementedError

class QBot:
    """The interface that q-bots implement. The actual Bot instance can
    be created through some type of qbot controller"""

    @property
    def cutoff(self) -> int:
        """Returns the number of moves after which we clip the reward, for easier online
        learning"""
        raise NotImplementedError

    @property
    def alpha(self) -> float:
        """Gets the exponential decay factor on the reward for moves. For a transition
        from game state (0) to 1, the first reward is not reduced, the following is reduced
        by alpha, the next by alpha^2, all the way up to the cutoff.

        Should be between 0 and 1 (exclusive)
        """
        raise NotImplementedError

    def evaluate(self, game_state: GameState, move: Move) -> float:
        """Evaluate the given move and return the anticipated (diminished)
        rewards"""
        raise NotImplementedError

    def learn(self, game_state: GameState, move: Move, reward: float) -> None:
        """Teach the underling model that it should have predicted the specified
        diminished reward from making the specified move in the given game state
        """
        raise NotImplementedError

    def think(self, max_time: float) -> None:
        """Can be used to kill time"""
        pass

    def save(self) -> None:
        """Should save the model"""
        raise NotImplementedError

class QBotMoveSelectionStyle(enum.IntEnum):
    """The various ways that a QBot could select its moves"""
    Greedy = 1 # always select the best move
    Stochastic = 2 # select moves according to likelihood

class QBotController(Bot):
    """The bot instance that is delegating to a single QBot instance. Optionally
    supports some percentage of teacher forcing, wherein the moves are replaced
    with a different bots moves some percentage of the time.

    Attributes:
        qbot (QBot): the qbot which is capable of evaluating moves
        rewarder (Rewarder): the thing that rewards the QBot
        moves (list[Move]): the moves this bot supports

        move_selstyle (QBotMoveSelectionStyle): how we determine what moves to take
            based on the predicted rewards

        teacher (Bot, optional): if specified, the "teacher" bot whose moves we adopt
            some percentage of the time
        teacher_force_amt (float, optional): if specified, value between 0 and 1 for
            the amount of teacher forcing to do

        save_interval (float): seconds between saves
        last_save (float): last time we saved the model

        gstate_cache (GameStateCache): the cache for game states.
        history (deque[(int, move)]): The left-most entries are the oldest and the
            rightmost entries are the newest.
    """
    def __init__(self, entity_iden: int, qbot: QBot, rewarder: Rewarder,
                 moves: typing.List[Move],
                 move_selstyle: QBotMoveSelectionStyle = QBotMoveSelectionStyle.Greedy,
                 teacher: Bot = None, teacher_force_amt: float = None,
                 gstate_cache: GameStateCache = None,
                 save_interval=30.0):
        super().__init__(entity_iden)
        self.qbot = qbot
        self.rewarder = rewarder
        self.moves = moves
        self.move_selstyle = move_selstyle
        self.teacher = teacher
        self.teacher_force_amt = teacher_force_amt

        if gstate_cache is None:
            gstate_cache = GameStateCache()
        self.gstate_cache = gstate_cache
        self.save_interval = save_interval
        self.last_save = time.time()
        self.history = deque()

    def move(self, game_state: GameState) -> Move:
        if self.teacher_force_amt and random.random() < self.teacher_force_amt:
            return self.teacher.move(game_state)

        move_rewards = [self.qbot.evaluate(game_state, move) for move in self.moves]

        if self.move_selstyle == QBotMoveSelectionStyle.Greedy:
            best_ind = int(np.argmax(move_rewards))
            return self.moves[best_ind]
        elif self.move_selstyle == QBotMoveSelectionStyle.Stochastic:
            softmaxed = torch.softmax(torch.tensor(move_rewards, dtype=torch.float), dim=0).numpy()
            return Move(np.random.choice(self.moves, p=softmaxed))
        else:
            raise NotImplementedError(f'unknown move sel style {self.move_selstyle}')

    def on_move(self, game_state: GameState, move: Move) -> None:
        self.gstate_cache.put(game_state)

        self.history.append((game_state.tick, move))
        if len(self.history) >= self.qbot.cutoff + 1:
            self._teach()

    def think(self, max_time: float):
        self.qbot.think(max_time)

    def _teach(self):
        reward = 0
        factor = 1
        for i in range(1, self.qbot.cutoff + 1):
            reward += factor * self.rewarder.reward(
                self.entity_iden,
                self.gstate_cache.fetch(self.history[i - 1][0]),
                self.gstate_cache.fetch(self.history[i][0])
            )
            factor *= self.qbot.alpha

        oldest = self.history.popleft()
        self.qbot.learn(self.gstate_cache.pop(oldest[0]), oldest[1], reward)

        if time.time() - self.last_save >= self.save_interval:
            self.qbot.save()
            self.last_save = time.time()

    def finished(self, game_state: GameState, result: UpdateResult) -> None:
        self.qbot.save()
