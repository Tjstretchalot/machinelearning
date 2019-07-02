"""A temporal difference learner with a multi-layer network with batch normalization as a nonlinear
function approximator, which learns offline through a large replay buffer."""

import os
import torch
import typing
import json
import numpy as np
import scipy.io
import datetime

from shared.teachers import NetworkTeacher, Network
from shared.models.ff import FeedforwardNetwork, FFTeacher, FFHiddenActivations
from shared.layers.norm import EvaluatingAbsoluteNormLayer, LearningAbsoluteNormLayer
import shared.trainer as tnr
import shared.perf_stats as perf_stats
import shared.typeutils as tus
import shared.cp_utils as cp_utils
import shared.filetools as filetools

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

def init_encoder(entity_iden):
    """Create an instance of the encoder for this model attached to the given entity"""
    return encoders.MergedFlatEncoders(
        (
            encoders.MoveEncoder(MOVE_MAP),
            encoders.LocationEncoder(entity_iden),
            encoders.StaircaseLocationEncoder(entity_iden),
        )
    )

ENCODE_DIM = init_encoder(None).dim
HIDDEN_DIM = 50

def _noop(*args, **kwargs):

    pass

def forward_with(fc_layers, norms, learning, inp, acts_cb):
    """Applies the forward pass to the given input data using the given
    fully connected layers and norms. If learning is set, it is invoked
    prior to the norms.

    Args:
        fc_layers (list[nn.Linear]): the three linear layers in the order they are invoked
        norms (list[nn.BatchNorm1d]): Or equivalent callables.
        learning (list[LearningAbsoluteNormLayer], optional): learning absolute norms or None
        inp (torch.tensor[batch_size, encode_dim]): the encoded game state
        acts_cb (callable, optional): passed FFHiddenActivations 4 times at times 0-3 respectively
    """
    tus.check_tensors(inp=(inp, (('batch', None), ('input_dim', ENCODE_DIM)), torch.float32))
    if acts_cb:
        tus.check_callable(acts_cb=acts_cb)
        acts_cb(FFHiddenActivations(layer=0, hidden_acts=inp))

    if learning:
        learning[0](inp)
    res = norms[0](inp)
    res = fc_layers[0](res)
    res = torch.tanh(res)
    if learning:
        learning[1](res)
    res = norms[1](res)
    if acts_cb:
        acts_cb(FFHiddenActivations(layer=1, hidden_acts=res))
    res = fc_layers[1](res)
    res = torch.tanh(res)
    if learning:
        learning[2](res)
    res = norms[2](res)
    if acts_cb:
        acts_cb(FFHiddenActivations(layer=2, hidden_acts=res))
    res = fc_layers[2](res)
    res = torch.tanh(res)
    if acts_cb:
        acts_cb(FFHiddenActivations(layer=3, hidden_acts=res))
    return res

class Deep1ModelTrain(FeedforwardNetwork):
    """The deep1 model we use for training. This is a feed forward network with the following shape:

    input (ENCODE_DIM)
    batch_norm
    linear (ENCODE_DIM -> HIDDEN_DIM)
    tanh
    batch_norm
    linear (HIDDEN_DIM -> HIDDEN_DIM)
    tanh
    batch_norm
    linear (HIDDEN_DIM -> 1)
    tanh

    This network can be swapped to Deep1ModelEval, which is the same except the batch norms are
    fixed to assume a specific mean and variance. The means and variances will be calculated
    using the Deep1ModelEval class.

    Attributes:
        fc_layers (list[nn.Linear]): the linear layers in the order that they are used
        bnorms (list[nn.BatchNorm1d]): the batch norms in the order that they are used
    """
    def __init__(self, fc_layers: typing.List[torch.nn.Linear]):
        super().__init__(ENCODE_DIM, 1, 3)

        self.fc_layers = fc_layers
        self.bnorms = [
            torch.nn.BatchNorm1d(ENCODE_DIM, affine=False, track_running_stats=False),
            torch.nn.BatchNorm1d(HIDDEN_DIM, affine=False, track_running_stats=False),
            torch.nn.BatchNorm1d(HIDDEN_DIM, affine=False, track_running_stats=False)
        ]

        for i, lyr in enumerate(self.fc_layers):
            self.add_module(f'layer{i}', lyr)

    def forward(self, inp: torch.tensor, acts_cb: typing.Callable = None): # pylint: disable=arguments-differ
        return forward_with(self.fc_layers, self.bnorms, None, inp, acts_cb)

    def save(self, outpath: str, exist_ok: bool = False):
        """Saves this network to the given path. The path should be a folder, because inside
        the folder we will store:
            model.pt - an equivalent network which can be shared and loaded with just pytorch
            layers.npz - the fully connected layer weights and biases in numpy form
                Each fully connected layer i has its weights stored in lyr_weight_i and its biases
                stored in lyr_bias_i
            layers.mat - the fully connected layer weights and biases in matlab form
                Same variable names as layers.npz
            readme.txt - stores relevant documentation for loading this model
        """
        if os.path.exists(outpath):
            if not exist_ok:
                raise FileExistsError(outpath)
            if not os.path.isdir(outpath):
                raise ValueError(f'expected outpath is directory, got {outpath} (not isdir)')
            filetools.deldir(outpath)

        os.makedirs(outpath)

        equiv_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(ENCODE_DIM, affine=False, track_running_stats=False),
            cp_utils.copy_linear(self.fc_layers[0]),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(HIDDEN_DIM, affine=False, track_running_stats=False),
            cp_utils.copy_linear(self.fc_layers[1]),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(HIDDEN_DIM, affine=False, track_running_stats=False),
            cp_utils.copy_linear(self.fc_layers[2]),
            torch.nn.Tanh()
        )
        torch.save(equiv_net, os.path.join(outpath, 'model.pt'))
        del equiv_net

        layer_data = {}
        for i, lyr in enumerate(self.fc_layers):
            layer_data[f'lyr_weight_{i}'] = lyr.weight.data.clone().numpy()
            layer_data[f'lyr_bias_{i}'] = lyr.bias.data.clone().numpy()
        np.savez_compressed(os.path.join(outpath, 'layers.npz'), **layer_data)
        scipy.io.savemat(os.path.join(outpath, 'layers.mat'), layer_data)

        with open(os.path.join(outpath, 'readme.txt'), 'w') as outfile:
            print('Model: Deep1ModelTrain', file=outfile)
            print(f'Date: {datetime.datetime.now()}', file=outfile)
            print('Constants:', file=outfile)
            for nm, const in {'ALPHA': ALPHA, 'CUTOFF': CUTOFF,
                              'ENCODE_DIM': ENCODE_DIM,
                              'HIDDEN_DIM': HIDDEN_DIM}.items():
                print(f'  {nm}: {const}', file=outfile)
            print('Class Documentation:', file=outfile)
            print(Deep1ModelTrain.__doc__, file=outfile)
            print(file=outfile)
            print('Function Documentation: ', file=outfile)
            print(Deep1ModelTrain.save.__doc__, file=outfile)

    @classmethod
    def load(cls, inpath: str) -> 'Deep1ModelTrain':
        """Loads the model that is stored in the given folder. It should have been stored there
        as if by save(inpath)."""
        tus.check(inpath=(inpath, str))
        if not os.path.exists(inpath):
            raise FileNotFoundError(inpath)
        lyr_data_path = os.path.join(inpath, 'layers.npz')
        if not os.path.exists(lyr_data_path):
            raise FileNotFoundError(lyr_data_path)

        fc_layers = []
        with np.load(lyr_data_path) as lyr_data:
            for i in range(3):
                weights_np = lyr_data[f'lyr_weight_{i}']
                bias_np = lyr_data[f'lyr_bias_{i}']
                lin_lyr = torch.nn.Linear(weights_np.shape[1], weights_np.shape[0])
                lin_lyr.weight.data[:] = torch.from_numpy(weights_np)
                lin_lyr.bias.data[:] = torch.from_numpy(bias_np)
                fc_layers.append(lin_lyr)
        return cls(fc_layers)

    @classmethod
    def create(cls) -> 'Deep1ModelTrain':
        """Creates a new random instance of this model"""
        return cls([
            torch.nn.Linear(ENCODE_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, 1)
        ])

class Deep1ModelEval(FeedforwardNetwork):
    """The deep1 model we can use for evaluating and as a target network. Instead of using the
    batch as the calculation for mean and variance, this uses a fixed value for each feature
    that is acquired in a earlier, potentially larger sample. This ensures we can evaluate the
    network effectively on the very first tick of a game, for example.

    Attributes:
        fc_layers (list[nn.Linear]): the same fully connected layers as the train model
        anorms (list[EvaluatingAbsoluteNormLayer]): replaces the batch norm layers
    """
    def __init__(self, fc_layers: typing.List[torch.nn.Linear],
                 anorms: typing.List[EvaluatingAbsoluteNormLayer]):
        super().__init__(ENCODE_DIM, 1, 3)
        self.fc_layers = fc_layers
        self.anorms = anorms

    def forward(self, inp: torch.tensor, acts_cb: typing.Callable = None): # pylint: disable=arguments-differ
        return forward_with(self.fc_layers, self.anorms, None, inp, acts_cb)

    def save(self, outpath: str, exist_ok: bool = False):
        """Saves this model to the given outpath. The outpath should be a folder. This will
        store the following things:
            model.pt - an equivalent network which can be loaded with just pytorch. This is
                accomplished by replacing the more memory efficient EvaluatingAbsoluteNormLayer
                with a linear layer with lots of 0s
            layers.npz - stores the fully connected layers with lyr_weight_i and lyr_bias_i, storing
                the norms in norm_means_i and norm_inv_std_i, where norm_inv_std_i is 1/std(feature)
            layers.mat - stores the equivalent data as layers.npz in matlab format
            readme.txt - relevant documentation for loading the model
        """
        if os.path.exists(outpath):
            if not exist_ok:
                raise FileExistsError(outpath)
            if not os.path.isdir(outpath):
                raise ValueError(f'expected outpath is dir, got {outpath} (not isdir)')
            filetools.deldir(outpath)

        os.makedirs(outpath)
        equiv_net = torch.nn.Sequential(
            self.anorms[0].to_linear(),
            cp_utils.copy_linear(self.fc_layers[0]),
            torch.nn.Tanh(),
            self.anorms[1].to_linear(),
            cp_utils.copy_linear(self.fc_layers[1]),
            torch.nn.Tanh(),
            self.anorms[2].to_linear(),
            cp_utils.copy_linear(self.fc_layers[2]),
            torch.nn.Tanh()
        )
        torch.save(equiv_net, os.path.join(outpath, 'model.pt'))
        del equiv_net

        layers = {}
        for i, norm in enumerate(self.anorms):
            layers[f'norm_means_{i}'] = norm.means.clone().numpy()
            layers[f'norm_inv_std_{i}'] = norm.inv_std.clone().numpy()
        for i, lyr in enumerate(self.fc_layers):
            layers[f'lyr_weight_{i}'] = lyr.weight.data.clone().numpy()
            layers[f'lyr_bias_{i}'] = lyr.bias.data.clone().numpy()
        np.savez_compressed(os.path.join(outpath, 'layers.npz'), **layers)
        scipy.io.savemat(os.path.join(outpath, 'layers.mat'), layers)

        with open(os.path.join(outpath, 'readme.txt'), 'w') as outfile:
            print('Model: Deep1ModelEval', file=outfile)
            print(f'Date: {datetime.datetime.now()}', file=outfile)
            print('Constants:', file=outfile)
            for nm, const in {'ALPHA': ALPHA, 'CUTOFF': CUTOFF,
                              'ENCODE_DIM': ENCODE_DIM,
                              'HIDDEN_DIM': HIDDEN_DIM}.items():
                print(f'  {nm}: {const}', file=outfile)
            print('Class Documentation:', file=outfile)
            print(Deep1ModelEval.__doc__, file=outfile)
            print(file=outfile)
            print('Function Documentation: ', file=outfile)
            print(Deep1ModelEval.save.__doc__, file=outfile)

    @classmethod
    def load(cls, inpath: str) -> 'Deep1ModelEval':
        """Loads the model in the specified directory, where the directory exists as if it was
        created by a call to save(inpath).
        """
        if not os.path.exists(inpath):
            raise FileNotFoundError(inpath)
        lyr_data_path = os.path.join(inpath, 'layers.npz')
        if not os.path.exists(lyr_data_path):
            raise FileNotFoundError(lyr_data_path)

        fc_layers = []
        anorms = []
        with np.load(lyr_data_path) as lyr_data:
            for i in range(3):
                weights_np = lyr_data[f'lyr_weight_{i}']
                bias_np = lyr_data[f'lyr_bias_{i}']
                lin_lyr = torch.nn.Linear(weights_np.shape[1], weights_np.shape[0])
                lin_lyr.weight.data[:] = torch.from_numpy(weights_np)
                lin_lyr.bias.data[:] = torch.from_numpy(bias_np)
                fc_layers.append(lin_lyr)

                means_np = lyr_data[f'norm_means_{i}']
                inv_std_np = lyr_data[f'norm_inv_std_{i}']
                anorms.append(EvaluatingAbsoluteNormLayer(
                    means_np.shape[0], torch.from_numpy(means_np), torch.from_numpy(inv_std_np)))

        return cls(fc_layers, anorms)

class Deep1ModelToEval(FeedforwardNetwork):
    """A deep1 model that was using a batch norm but is in the process of being converted to an
    absolute norm.

    Attributes:
        fc_layers (list[nn.Linear]): the fully connected layers that we learned
        cur_norms (list[union[BatchNorm1d, EvaluatingAbsoluteNormLayer]]): the current norm we use
        learning (list[LearningAbsoluteNormLayer]): the new approximation for the norms
    """
    def __init__(self, fc_layers: typing.List[torch.nn.Linear]):
        super().__init__(ENCODE_DIM, 1, 3)
        self.fc_layers = fc_layers
        self.cur_norms = [
            torch.nn.BatchNorm1d(ENCODE_DIM, affine=False, track_running_stats=False),
            torch.nn.BatchNorm1d(HIDDEN_DIM, affine=False, track_running_stats=False),
            torch.nn.BatchNorm1d(HIDDEN_DIM, affine=False, track_running_stats=False),
        ]
        self.learning = [
            LearningAbsoluteNormLayer(ENCODE_DIM),
            LearningAbsoluteNormLayer(HIDDEN_DIM),
            LearningAbsoluteNormLayer(HIDDEN_DIM)
        ]

    def forward(self, inp: torch.tensor, acts_cb: typing.Callable = None): # pylint: disable=arguments-differ
        return forward_with(self.fc_layers, self.cur_norms, self.learning, inp, acts_cb)

    def learning_to_current(self):
        """Replaces the current norms with the learned norms and resets the
        learning norms.
        """
        self.cur_norms = [learn.to_evaluative(True) for learn in self.learning]
        self.learning = [
            LearningAbsoluteNormLayer(ENCODE_DIM),
            LearningAbsoluteNormLayer(HIDDEN_DIM),
            LearningAbsoluteNormLayer(HIDDEN_DIM)
        ]

    def to_evaluative(self) -> Deep1ModelEval:
        """Uses the current norm layers to create a corresponding evaluation model"""
        if not isinstance(self.cur_norms[0], EvaluatingAbsoluteNormLayer):
            raise ValueError('learning_to_current must be called at least once')

        return Deep1ModelEval(self.fc_layers, self.cur_norms)

def _init_model():
    if os.path.exists(EVAL_MODELFILE):
        raise FileExistsError(EVAL_MODELFILE)

    train_model = Deep1ModelTrain.create()
    train_model.save(MODELFILE)
    model = Deep1ModelEval(train_model.fc_layers, [
        EvaluatingAbsoluteNormLayer.create_identity(ENCODE_DIM),
        EvaluatingAbsoluteNormLayer.create_identity(HIDDEN_DIM),
        EvaluatingAbsoluteNormLayer.create_identity(HIDDEN_DIM)
    ])
    model.save(EVAL_MODELFILE)
    return model

SAVEDIR = os.path.join('out', 'or_reinforce', 'deep', 'deep1')
MODELFILE = os.path.join(SAVEDIR, 'model')
EVAL_MODELFILE = os.path.join(SAVEDIR, 'model_eval')
REPLAY_FOLDER = os.path.join(SAVEDIR, 'replay')
REPLAY_FOLDER_2 = os.path.join(SAVEDIR, 'replay_tmp')
SETTINGS_FILE = os.path.join(SAVEDIR, 'settings.json')

class DeepQBot(qbot.QBot):
    """The Q-bot implementation

    Attributes:
        entity_iden (int): the entity we are controlling
        model (FeedforwardComplex): the model that does the evaluating
        teacher (FFTeacher): the teacher for the model
        evaluation (bool): True to not store experiences, False to store experiences

        replay (WritableReplayBuffer, optional): the buffer for replays

        encoder (Encoder): the encoder
    """
    def __init__(self, entity_iden: int, replay_path=REPLAY_FOLDER, evaluation=False):
        self.entity_iden = entity_iden
        if not os.path.exists(EVAL_MODELFILE):
            _init_model()

        self.model = Deep1ModelEval.load(EVAL_MODELFILE)

        self.teacher = FFTeacher()
        self.evaluation = evaluation
        self.encoder = init_encoder(entity_iden)

        if not evaluation:
            self.replay = replay_buffer.FileWritableReplayBuffer(replay_path, exist_ok=True)
        else:
            self.replay = None

    def __call__(self, entity_iden):
        self.entity_iden = entity_iden
        self.encoder = init_encoder(entity_iden)

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

def deep1(entity_iden: int, settings: str = None) -> 'Bot':  # noqa: F821
    """Creates a new deep1 bot"""
    if not settings:
        settings = SETTINGS_FILE
    if not os.path.exists(settings):
        raise FileNotFoundError(settings)

    with open(settings, 'r') as infile:
        settings = json.load(infile)

    return MyQBotController(
        entity_iden,
        DeepQBot(entity_iden, settings['replay_path'], (('eval' in settings) and settings['eval'])),
        rewarders.SCRewarder(bigreward=(1 - ALPHA)),  # biggest cumulative reward is 1
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
            self.encoders_by_id[ent_id] = init_encoder(ent_id)

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
                self.encoders_by_id[ent_id] = init_encoder(ent_id)

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

    replay_buffer.balance_experiences(REPLAY_FOLDER, [replay_buffer.PositiveExperience(), replay_buffer.NegativeExperience()])
    replay = replay_buffer.FileReadableReplayBuffer(REPLAY_FOLDER, perf=perf)
    try:
        print(f'loaded {len(replay)} experiences for replay...')
        if not os.path.exists(MODELFILE):
            _init_model()

        network = Deep1ModelTrain.load(MODELFILE)
        teacher = MyTeacher(FFTeacher())

        train_pwl = MyPWL(replay, Deep1ModelEval.load(EVAL_MODELFILE), teacher)
        test_pwl = train_pwl

        def update_target(ctx: tnr.GenericTrainingContext, hint: str):
            ctx.logger.info('swapping target network, hint=%s', hint)
            network.save(MODELFILE, exist_ok=True)

            new_target = Deep1ModelToEval(network.fc_layers)
            for _ in range(3):
                train_pwl.mark()
                for _ in range(0, 1024, ctx.batch_size):
                    train_pwl.fill(ctx.points, ctx.labels)
                    teacher.classify_many(new_target, ctx.points, ctx.labels.unsqueeze(1))
                new_target.learning_to_current()
                train_pwl.reset()

            new_target = new_target.to_evaluative()
            new_target.save(EVAL_MODELFILE, exist_ok=True)

            train_pwl.target_model = new_target

        trainer = tnr.GenericTrainer(
            train_pwl=train_pwl,
            test_pwl=test_pwl,
            teacher=teacher,
            batch_size=32,
            learning_rate=0.0001,
            optimizer=torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.0001),
            criterion=torch.nn.MSELoss()
        )
        (trainer
        .reg(tnr.EpochsTracker())
        .reg(tnr.EpochsStopper(100))
        .reg(tnr.InfOrNANDetecter())
        .reg(tnr.InfOrNANStopper())
        .reg(tnr.DecayTracker())
        .reg(tnr.DecayStopper(1))
        .reg(tnr.OnEpochCaller.create_every(update_target, skip=CUTOFF)) # smaller cutoffs require more bootstrapping
        .reg(tnr.DecayOnPlateau())
        )
        res = trainer.train(network, target_dtype=torch.float32, point_dtype=torch.float32, perf=perf)
        if res['inf_or_nan']:
            print('training failed! inf or nan!')
    finally:
        replay.close()

if __name__ == '__main__':
    offline_learning()
