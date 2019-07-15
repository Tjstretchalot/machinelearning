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
import os

from shared.measures.utils import NetworkHiddenActivations, StackedNetworkActivations
from shared.filetools import deldir

from optimax_rogue_bots.bot import Bot
from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
from optimax_rogue.logic.updater import UpdateResult
from optimax_rogue_bots.gui.state_action_bot import StateActionBot
import optimax_rogue_bots.gui.packets as gpackets

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

    def started(self, game_state: GameState) -> None:
        """Called when we first connect"""
        pass

    def evaluate(self, game_state: GameState, move: Move) -> float:
        """Evaluate the given move and return the anticipated (diminished)
        rewards"""
        raise NotImplementedError

    def evaluate_all(self, game_state: GameState,
                     moves: typing.List[Move]) -> typing.Iterable[float]:
        """Evaluates all the moves available to this bot and returns them in the order
        that the QBotController moves was initialized to. The moves is provided for
        generic implementations, but can be assumed to be the same as the move map.
        """
        return [self.evaluate(game_state, move) for move in moves]

    def learn(self, game_state: GameState, move: Move, new_state: GameState,
              reward_raw: float, reward_pred: float) -> None:
        """Teach the underlying model that it should have predicted the specified
        diminished reward from making the specified move in the given game state

        Arguments:
            game_state (GameState): the state of the game when we made the given move
            move (Move): the move that we made
            new_state (GameState): the state of the game we ended up in after cutoff moves
            reward_raw (float): the reward that we actually got from the cutoff timesteps
                after the reward (cumulative and discounted)
            reward_pred (float): the reward that we predict that we will get in the state
                that we ended up to (discounted appropriately)
        """
        raise NotImplementedError

    def think(self, max_time: float) -> None:
        """Can be used to kill time"""
        pass

    def save(self) -> None:
        """Should save the model"""
        raise NotImplementedError

class HiddenStateQBot:
    """An optional interface for the QBot that exposes its hidden state"""

    def get_hidden(self, game_state: GameState, move: Move) -> NetworkHiddenActivations:
        """Returns the hidden activations of the qbot as it evaluates the specified game
        state and move. The activations may be recurrent or feedforward or some combination,
        but it must be consistent (i.e., always have the same number of activations and each
        activation block is the same size through different state-action pairs)
        """
        raise NotImplementedError

class QBotMoveSelectionStyle(enum.IntEnum):
    """The various ways that a QBot could select its moves"""
    Greedy = 1 # always select the best move
    Stochastic = 2 # select moves according to likelihood

class QBotController(StateActionBot):
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

        frozen (bool): true if learning is not occuring right now, false otherwise. Should
            be treated as read-only since freezing will stop the history from being updated

        save_activations (bool): if the activations of the network should be saved. the network
            must also implemenet HiddenStateQBot. Should be treated as readonly
        activations_folder (str, optional): if save_activations is True, this is the folder to
            which we save activations.
        activations_per_block (int, optional): the maximum number of activations that we store in
            memory. After reaching this number of activations in memory, they are dumped to file.
        activations (StackedNetworkActivations, optional): the in-memory activations that we
            currently have
        activations_block (int, optional): the current block we are on, nonnegative. Determines
            where we next save

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
                 save_interval=30.0, frozen=False,
                 save_activations: bool = False, activations_folder: str = None,
                 overwrite_activations: bool = False,
                 activations_per_block: int = 1000):
        super().__init__(entity_iden)
        self.qbot = qbot
        self.rewarder = rewarder
        self.moves = moves
        self.move_selstyle = move_selstyle
        self.teacher = teacher
        self.teacher_force_amt = teacher_force_amt
        self.frozen = frozen

        if save_activations:
            if not isinstance(activations_folder, str):
                raise ValueError('save_activations is set but activations_folder is '
                                 + f'{activations_folder} (str expected, got '
                                 + f'{type(activations_folder)})')
            if not isinstance(qbot, HiddenStateQBot):
                raise ValueError(f'save_activations is True but qbot {type(qbot)} does '
                                 + 'not expose hidden acts')

        self.save_activations = save_activations
        self.activations_folder = activations_folder
        self.activations_per_block = activations_per_block
        self.activations = (
            None
            if not self.save_activations
            else StackedNetworkActivations(activations_per_block)
        )
        self.activations_block = None
        if self.save_activations:
            self.activations_block = 1
            if not overwrite_activations:
                while os.path.exists(os.path.join(self.activations_folder,
                                                  str(self.activations_block))):
                    self.activations_block += 1

        if gstate_cache is None:
            gstate_cache = GameStateCache()
        self.gstate_cache = gstate_cache
        self.save_interval = save_interval
        self.last_save = time.time()
        self.history = deque()

    def __call__(self, entity_iden):
        self.entity_iden = entity_iden
        self.qbot(entity_iden)
        return self

    @classmethod
    def scale_style(cls) -> gpackets.SetScaleStylePacket:
        """Returns the scale style packet that the bot prefers"""
        return gpackets.SetScaleStylePacket(gpackets.ScaleStyle.TemperatureSoftArgMax, 0.02)

    @classmethod
    def pitch(cls):
        import warnings
        warnings.warn('Should override pitch!', category=UserWarning)
        return (
            'QBotController',
            'This bot is missing a description. To add a description, subclass QBotController '
            + 'and override the classmethod pitch'
        )

    @classmethod
    def supported_moves(cls):
        import warnings
        warnings.warn('Should override supported_moves!', category=UserWarning)
        return [Move.Left, Move.Right, Move.Up, Move.Down]

    def started(self, game_state: GameState) -> Move:
        self.qbot.started(game_state)

    def move(self, game_state: GameState) -> Move:
        if self.teacher_force_amt and random.random() < self.teacher_force_amt:
            return self.teacher.move(game_state)

        move_rewards = [float(pred) for pred in self.qbot.evaluate_all(game_state, self.moves)]
        if self.move_selstyle == QBotMoveSelectionStyle.Greedy:
            best_ind = int(np.argmax(move_rewards))
            return self.moves[best_ind]
        elif self.move_selstyle == QBotMoveSelectionStyle.Stochastic:
            softmaxed = torch.softmax(torch.tensor(move_rewards, dtype=torch.float), dim=0).numpy()
            return Move(np.random.choice(self.moves, p=softmaxed))
        else:
            raise NotImplementedError(f'unknown move sel style {self.move_selstyle}')

    def evaluate(self, game_state: GameState, move: Move) -> float:
        return self.qbot.evaluate(game_state, move)

    def on_move(self, game_state: GameState, move: Move) -> None:
        self.gstate_cache.put(game_state)

        if self.save_activations:
            qbot: HiddenStateQBot = self.qbot
            acts = qbot.get_hidden(game_state, move)
            self.activations.append_acts(acts)
            if self.activations.num_pts == self.activations_per_block:
                save_folder = os.path.join(self.activations_folder, self.activations_block)
                if os.path.exists(save_folder):
                    deldir(save_folder)

                self.activations.save(save_folder)
                self.activations_block += 1
                self.activations.num_pts = 0

        if not self.frozen:
            self.history.append((game_state.tick, move))
            if len(self.history) >= self.qbot.cutoff + 1:
                self._teach()

    def _det_move(self, game_state: GameState) -> Move:
        """Deterministic greedy movement without teacher forcing"""
        move_rewards = self.qbot.evaluate_all(game_state, self.moves)
        if isinstance(move_rewards, (tuple, list, np.ndarray)):
            best_ind = int(np.argmax(move_rewards))
        else:
            best_ind = move_rewards.argmax().item()
        return self.moves[best_ind]

    def think(self, max_time: float):
        if not self.frozen:
            self.qbot.think(max_time)

    def _teach(self):
        if self.frozen:
            return

        reward = 0
        factor = 1
        for i in range(1, self.qbot.cutoff + 1):
            reward += factor * self.rewarder.reward(
                self.entity_iden,
                self.gstate_cache.fetch(self.history[i - 1][0]),
                self.gstate_cache.fetch(self.history[i][0])
            )
            factor *= self.qbot.alpha

        latest_gamestate = self.gstate_cache.fetch(self.history[self.qbot.cutoff][0])
        reward_pred = factor * float(max(self.qbot.evaluate_all(latest_gamestate, self.moves)))

        latest = self.history[self.qbot.cutoff]
        latest_state = self.gstate_cache.fetch(latest[0])
        oldest = self.history.popleft()
        oldest_state = self.gstate_cache.pop(oldest[0])
        self.qbot.learn(
            oldest_state, oldest[1], latest_state, reward, reward_pred)

        if time.time() - self.last_save >= self.save_interval:
            self.qbot.save()
            self.last_save = time.time()

    def finished(self, game_state: GameState, result: UpdateResult) -> None:
        if self.frozen:
            return
        self.qbot.save()
