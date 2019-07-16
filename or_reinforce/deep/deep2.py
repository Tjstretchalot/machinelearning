"""This is another deep q network that uses a replay buffer and a target network for training,
which tries to address the following issues from the deep1 bot:
    1. Training instability: if the network every gets into a state where 3 actions have been
       pushed to have very negative expectations, the network will have a dataset which heavily
       biases a single move. This will tend to result in a negative reward for that move. The
       problem is the network can "solve" the predicted value now with just a large negative
       bias on the output layer.
    2. Related to 1, the network tends not to adequately take into account the move that is
       being performed.
    3. It is very difficult to bias the network towards moves that it has less experience with,
       because the weights of all the moves are heavily tied together.
    4. The implementation of the deep1 bot made it difficult to modify the network structure,
       which this addresses.

The solution that the deep2 bot uses is inspired by the deep convolutional networks leading into
a prediction for the Q-value of every possible move. These networks can be thought of as sharing
the same feature generation layers but using a different combination of features to assert the
output.

To train these networks one can use a special loss function, which is the constant 0 on the outputs
that are not known and an appropriate loss function (such as MSE) for the output that is known.
Under this construction it becomes possible to bias less-common moves upward by simply adding a
small loss that tends the other moves toward positive 1.
"""

import os
import torch
import typing
import json
import numpy as np

from shared.models.ff import FeedforwardNetwork, FFTeacher, FFHiddenActivations
from shared.layers.norm import EvaluatingAbsoluteNormLayer, LearningAbsoluteNormLayer
from shared.layers.affine import AffineLayer
import shared.nonlinearities as snonlins
import shared.trainer as tnr
import shared.perf_stats as perf_stats
import shared.typeutils as tus
import shared.filetools as filetools
import shared.measures.utils as mutils

import or_reinforce.utils.qbot as qbot
import or_reinforce.utils.rewarders as rewarders
import or_reinforce.utils.encoders as encoders
import or_reinforce.deep.replay_buffer as replay_buffer

from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
from optimax_rogue_bots.randombot import RandomBot

import shared.pwl as pwl


SAVEDIR = os.path.join('out', 'or_reinforce', 'deep', 'deep2')
MODELFILE = os.path.join(SAVEDIR, 'model')
EVAL_MODELFILE = os.path.join(SAVEDIR, 'model_eval')
REPLAY_FOLDER = os.path.join(SAVEDIR, 'replay')
SETTINGS_FILE = os.path.join(SAVEDIR, 'settings.json')

MOVE_MAP = [Move.Left, Move.Right, Move.Up, Move.Down]
MOVE_LOOKUP = dict((move, i) for i, move in enumerate(MOVE_MAP))
ALPHA = 0.8
CUTOFF = 1
PRED_WEIGHT = ALPHA ** CUTOFF

INVALID_REWARD = -2

def init_encoder(entity_iden):
    """Create an instance of the encoder for this model attached to the given entity"""
    return encoders.MergedFlatEncoders(
        [
            encoders.StaircaseDirectionOneHotEncoder(3, entity_iden),
            encoders.SurroundBarrierEncoder(entity_iden, 5),
            encoders.SurroundEntityEncoder(entity_iden, 3)
        ]
    )

ENCODE_DIM = init_encoder(None).dim
HIDDEN_DIM = 100
OUTPUT_DIM = len(MOVE_MAP)

class StackableLayer(torch.nn.Module):
    """An extraction layer within the deep2 network

    Attributes:
        fc (nn.Linear): the linear layer
        nonlin (callable): the nonlinearity
        nonlin_nm (str): the name for the nonlinearity (see shared.nonlinearities)
        learning (LearningAbsoluteNormLayer, optional): an optional learning
            normalization layer
        norm (BatchNorm1d or EvaluatingAbsoluteNormLayer): the actual normalization layer
        affine (AffineLayer): the affine transformation (decoupled from the norm)
    """
    def __init__(self, fc: torch.nn.Linear, nonlin_nm: str,
                 learning: typing.Optional[LearningAbsoluteNormLayer],
                 norm: typing.Union[torch.nn.BatchNorm1d, EvaluatingAbsoluteNormLayer],
                 affine: AffineLayer):
        super().__init__()
        self.fc = fc
        self.nonlin_nm = nonlin_nm
        self.nonlin = snonlins.extended_lookup(nonlin_nm)
        self.learning = learning
        self.norm = norm
        self.affine = affine

    @classmethod
    def create(cls, in_features: int, out_features: int, nonlin_nm='tanh'):
        """Creates a new random stackable layer using the default torch weight and bias
        initializations for the fully connected layer and an identity affine transformation"""
        return cls(
            torch.nn.Linear(in_features, out_features),
            nonlin_nm,
            None,
            torch.nn.BatchNorm1d(out_features, affine=False, track_running_stats=False),
            AffineLayer.create(out_features)
        )

    def identity_eval(self):
        """Swaps the normalization layers with identity layers"""
        if self.learning:
            raise ValueError('in learning mode')
        self.norm = EvaluatingAbsoluteNormLayer.create_identity(self.fc.out_features)

    def start_stat_tracking(self):
        """Inserts a learning layer prior to the normalization layer"""
        if self.learning:
            raise ValueError('already in learning mode')
        self.learning = LearningAbsoluteNormLayer(self.fc.out_features)

    def stat_tracking_to_norm(self):
        """Converts the learning layer to the norm layer"""
        if not self.learning:
            raise ValueError('not in learning mode')
        self.norm = self.learning.to_evaluative(True)
        self.learning = LearningAbsoluteNormLayer(self.fc.out_features)

    def stop_stat_tracking(self):
        """Removes the learning layer, continuing to use the absolute norm layer"""
        if not self.learning:
            raise ValueError('not in learning mode')
        self.learning = None

    def to_train(self):
        """Switches back to training mode after a call to stop_learning by switching the
        norm back to a batch norm"""
        if self.learning:
            raise ValueError('in learning mode')
        self.norm = torch.nn.BatchNorm1d(
            self.fc.out_features, affine=False, track_running_stats=False)

    def forward(self, inp: torch.tensor): # pylint: disable=arguments-differ
        acts = self.nonlin(self.fc(inp))
        if self.learning:
            self.learning(acts)
        return self.affine(self.norm(acts))

    def to_raw_numpy(self) -> typing.Tuple[
            typing.Dict[str, np.ndarray], typing.Dict[str, typing.Any]]:
        """Creates a complete representation of this stackable layer with a dict of
        numpy arrays for tabular data and a dict of primitives for metadata. This
        can be converted back into a stackable layer with from_raw_numpy.
        """
        if self.learning is not None:
            raise ValueError('cannot save while in learning mode')

        npres = {
            'fc_weights': self.fc.weight.data.numpy(),
            'fc_biases': self.fc.bias.data.numpy(),
            'affine_mult': self.affine.mult.data.numpy(),
            'affine_add': self.affine.add.data.numpy(),
        }

        primres = {
            'nonlin_nm': self.nonlin_nm
        }

        if isinstance(self.norm, torch.nn.BatchNorm1d):
            primres['norm_type'] = 'batch'
        else:
            primres['norm_type'] = 'absolute'
            npres['norm_inv_std'] = self.norm.inv_std.numpy()
            npres['norm_means'] = self.norm.means.numpy()

        return npres, primres

    @classmethod
    def from_raw_numpy(cls, npres, primres, prefix='') -> 'StackableLayer':
        """Loads the stackable layer stored with to_raw_numpy, with an optional
        prefix on the keys for the numpy values.

        Args:
            npres (dict[str, np.ndarray]): the numpy values
            primres (dict[str, any]): the primitive values
            prefix (str, optional): the prefix for keys when fetching from npres
        """
        fc_weights = npres[prefix + 'fc_weights']
        fc = torch.nn.Linear(fc_weights.shape[1], fc_weights.shape[0])
        fc.weight.data[:] = torch.from_numpy(fc_weights)
        fc.bias.data[:] = torch.from_numpy(npres[prefix + 'fc_biases'])

        affine = AffineLayer(
            fc.out_features,
            torch.from_numpy(npres[prefix + 'affine_mult']),
            torch.from_numpy(npres[prefix + 'affine_add'])
        )

        nonlin_nm = primres['nonlin_nm']

        norm = None
        if primres['norm_type'] == 'batch':
            norm = torch.nn.BatchNorm1d(fc.out_features, affine=False, track_running_stats=False)
        else:
            norm = EvaluatingAbsoluteNormLayer(
                fc.out_features,
                torch.from_numpy(npres[prefix + 'norm_means']),
                torch.from_numpy(npres[prefix + 'norm_inv_std'])
            )

        return cls(fc, nonlin_nm, None, norm, affine)

def _noop(*args, **kwargs):
    pass

class Deep2Network(FeedforwardNetwork):
    """The network for the deep2 bot. This consists of a network of the following shape:

    Normalization
    FC (encode to hidden)
    Nonlinearity
    FC (hidden to hidden) -|
    Nonlinearity           |                  FEATURE EXTRACTION
    Normalization          |
                          /
                    repeat (extraction depth) times
    FC (hidden to output) (no bias)
    Nonlinearity

    Attributes:
        inp_norm (union[BatchNorm1d, EvaluatingAbsoluteLayer]): input normalization
        inp_learning (LearningAbsoluteLayer, optional): learning input normalization
        inp_affine (AffineLayer): the input affine layer
        inp_layer (StackableLayer): the input layer
        extract_layers (list[StackableLayer]): the feature extraction layers
        out_layer (torch.nn.Linear): the output layer
        out_nonlin_nm (str): the name for the output nonlinearity (see shared.nonlinearites)
        out_nonlin (typing.Callable): the output nonlinearity
    """
    def __init__(self, inp_norm: typing.Union[torch.nn.BatchNorm1d, EvaluatingAbsoluteNormLayer],
                 inp_learning: typing.Optional[LearningAbsoluteNormLayer], inp_affine: AffineLayer,
                 inp_layer: StackableLayer,
                 extract_layers: typing.List[StackableLayer],
                 out_layer: torch.nn.Linear,
                 out_nonlin_nm: str):
        super().__init__(ENCODE_DIM, OUTPUT_DIM, 2 + len(extract_layers))
        self.inp_norm = inp_norm
        self.inp_learning = inp_learning
        self.inp_affine = inp_affine
        self.inp_layer = inp_layer
        self.extract_layers = extract_layers
        self.out_layer = out_layer
        self.out_nonlin_nm = out_nonlin_nm
        self.out_nonlin = snonlins.extended_lookup(out_nonlin_nm)

        for i, lyr in enumerate(self.extract_layers):
            self.add_module(f'extract_lyr_{i}', lyr)

    @classmethod
    def create(cls):
        """Creates a new instance of the deep2 network in the batchnorm mode"""
        return cls(
            torch.nn.BatchNorm1d(ENCODE_DIM, affine=False, track_running_stats=False),
            None,
            AffineLayer.create(ENCODE_DIM),
            StackableLayer.create(ENCODE_DIM, HIDDEN_DIM),
            [StackableLayer.create(HIDDEN_DIM, HIDDEN_DIM) for i in range(5)],
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False),
            'tanh'
        )

    def identity_eval(self) -> None:
        """Swaps the normalization layers with identity layers. Required when you dont have any
        data to fetch statistics on but you can't use batch norms"""
        self.inp_norm = EvaluatingAbsoluteNormLayer.create_identity(ENCODE_DIM)
        self.inp_layer.identity_eval()
        for lyr in self.extract_layers:
            lyr.identity_eval()

    def start_stat_tracking(self) -> None:
        """Begins switching this network to evaluation mode. In evaluation mode there are no
        batch norm layers - they are instead replaced with absolute normalization layers. This
        inserts learning norm layers prior to the batch norm layers, which can be converted
        to absolute layers after this network has been shown some samples"""
        self.inp_learning = LearningAbsoluteNormLayer(ENCODE_DIM)
        self.inp_layer.start_stat_tracking()
        for lyr in self.extract_layers:
            lyr.start_stat_tracking()

    def stat_tracking_to_norm(self):
        """Swaps the norm layers with the current learned norm layers and resets the
        learning layers"""
        self.inp_norm = self.inp_learning.to_evaluative(True)
        self.inp_learning = LearningAbsoluteNormLayer(ENCODE_DIM)
        self.inp_layer.stat_tracking_to_norm()
        for lyr in self.extract_layers:
            lyr.stat_tracking_to_norm()

    def stop_stat_tracking(self):
        """Stops tracking statistics between layers without changing the current norm"""
        self.inp_learning = None
        self.inp_layer.stop_stat_tracking()
        for lyr in self.extract_layers:
            lyr.stop_stat_tracking()

    def to_train(self):
        """Switches the norm layers to batch norm layers which are suitable for training"""
        self.inp_norm = torch.nn.BatchNorm1d(ENCODE_DIM, affine=False, track_running_stats=False)
        self.inp_layer.to_train()
        for lyr in self.extract_layers:
            lyr.to_train()

    def forward(self, inps: torch.tensor, acts_cb: typing.Callable = _noop) -> torch.tensor: # pylint: disable=arguments-differ
        acts_cb(FFHiddenActivations(layer=0, hidden_acts=inps))
        if self.inp_learning:
            self.inp_learning(inps)
        acts = self.inp_layer(self.inp_affine(self.inp_norm(inps)))
        acts_cb(FFHiddenActivations(layer=1, hidden_acts=acts))
        for i, lyr in enumerate(self.extract_layers):
            acts = lyr(acts)
            acts_cb(FFHiddenActivations(layer=i + 2, hidden_acts=acts))
        acts = self.out_nonlin(self.out_layer(acts))
        acts_cb(FFHiddenActivations(layer=2 + len(self.extract_layers), hidden_acts=acts))
        return acts

    def save(self, outpath: str, exist_ok: bool = False, compress: bool = False):
        """Saves this model to the given folder or archive. Must not be in the middle of
        learning the norm layers. The only valid archive format is a zip, and it does
        not need to have that extension appended to the outpath. Note that this method
        is much nicer than the default pickling approach by torch since it does not
        require this package to load the numpy file.

        Args:
            outpath (str): the folder or archive to save to
            exist_ok (bool): True to overwrite the folder/archive if it exists, False to raise
                an error if either exists. Default False.
            compress (bool): True to compress to a zip, False to leave as a folder. Default False
        """
        tus.check(outpath=(outpath, str))
        outpath, outpath_wo_ext = mutils.process_outfile(outpath, exist_ok, compress)

        npres = {}
        primres = {'num_extract': len(self.extract_layers)}

        npres['inp_affine_mult'] = self.inp_affine.mult.data.numpy()
        npres['inp_affine_add'] = self.inp_affine.add.data.numpy()

        if isinstance(self.inp_norm, EvaluatingAbsoluteNormLayer):
            primres['inp_norm_style'] = 'abs'
            npres['inp_norm_means'] = self.inp_norm.means.numpy()
            npres['inp_norm_inv_std'] = self.inp_norm.inv_std.numpy()
        else:
            primres['inp_norm_style'] = 'batch'

        lyr_npres, lyr_primres = self.inp_layer.to_raw_numpy()
        for k, v in lyr_npres.items():
            npres[f'inp_lyr_{k}'] = v
        primres['inp_lyr'] = lyr_primres

        for i, lyr in enumerate(self.extract_layers):
            lyr_npres, lyr_primres = lyr.to_raw_numpy()
            for k, v in lyr_npres.items():
                npres[f'extract_{i}_{k}'] = v
            primres[f'extract_{i}'] = lyr_primres

        npres['out_weights'] = self.out_layer.weight.data.numpy()

        primres['out_nonlin_nm'] = self.out_nonlin_nm

        if os.path.exists(outpath_wo_ext):
            filetools.deldir(outpath_wo_ext)

        os.makedirs(outpath_wo_ext)

        np.savez_compressed(os.path.join(outpath_wo_ext, 'np.npz'), **npres)
        with open(os.path.join(outpath_wo_ext, 'prims.json'), 'w') as outfile:
            json.dump(primres, outfile)

        if compress:
            if os.path.exists(outpath):
                os.remove(outpath)
            filetools.zipdir(outpath_wo_ext)

    @classmethod
    def load(cls, inpath: str, compress: bool = False) -> 'Deep2Network':
        """Loads this model from the given folder or archive, as if it was saved to it
        by save. Note that this will prefer the folder to the archive if both exist.

        Args:
            inpath (str): the same path that was passed to save, up to an optional archive
                extension.
            compress (bool): if the folder should be compressed for future use when we are
                done. Regardless of previous state, if compress is False then the result is
                that inpath is extracted when this function completes and if compress is True
                then inpath is compressed when this function completes.
        """
        tus.check(inpath=(inpath, str))

        inpath, inpath_wo_ext = mutils.process_outfile(inpath, True, False)

        if not os.path.exists(inpath_wo_ext):
            filetools.unzip(inpath)

        with open(os.path.join(inpath_wo_ext, 'prims.json'), 'r') as infile:
            prims = json.load(infile)

        num_extract = prims['num_extract']
        out_nonlin_nm = prims['out_nonlin_nm']
        tus.check(num_extract=(num_extract, int), out_nonlin_nm=(out_nonlin_nm, str))

        with np.load(os.path.join(inpath_wo_ext, 'np.npz')) as npres:
            inp_affine = AffineLayer(
                ENCODE_DIM,
                torch.from_numpy(npres['inp_affine_mult']),
                torch.from_numpy(npres['inp_affine_add'])
            )
            inp_norm = None
            if prims['inp_norm_style'] == 'abs':
                inp_norm = EvaluatingAbsoluteNormLayer(
                    ENCODE_DIM,
                    torch.from_numpy(npres['inp_norm_means']),
                    torch.from_numpy(npres['inp_norm_inv_std'])
                )
            else:
                inp_norm = torch.nn.BatchNorm1d(
                    ENCODE_DIM, affine=False, track_running_stats=False)

            inp = StackableLayer.from_raw_numpy(npres, prims['inp_lyr'], 'inp_lyr_')

            extract_layers = []
            for i in range(num_extract):
                extract_layers.append(
                    StackableLayer.from_raw_numpy(npres, prims[f'extract_{i}'], f'extract_{i}_')
                )

            out_weights = npres['out_weights']
            out_layer = torch.nn.Linear(out_weights.shape[1], out_weights.shape[0], bias=False)
            out_layer.weight.data[:] = torch.from_numpy(out_weights)

        if compress:
            filetools.zipdir(inpath_wo_ext)

        return cls(inp_norm, None, inp_affine, inp, extract_layers, out_layer, out_nonlin_nm)

def init_or_load_model(evaluation=False) -> Deep2Network:
    """Initializes the deep2 network model if the model does not exist in the standard location,
    otherwise creates a new one and saves it to the standard location"""
    if os.path.exists(MODELFILE) or os.path.exists(MODELFILE + '.zip'):
        return Deep2Network.load(EVAL_MODELFILE if evaluation else MODELFILE)
    model = Deep2Network.create()
    model.save(MODELFILE)

    eval_model = Deep2Network.load(MODELFILE)
    eval_model.identity_eval()
    eval_model.save(EVAL_MODELFILE)
    print(f'saved initial eval model to {EVAL_MODELFILE}')
    return eval_model if evaluation else model

class Deep2QBot(qbot.QBot):
    """The Q-bot implementation

    Attributes:
        entity_iden (int): the entity we are controlling
        model (Deep2Network): the model that does the evaluating
        teacher (FFTeacher): the teacher for the model
        evaluation (bool): True to not store experiences, False to store experiences

        replay (WritableReplayBuffer, optional): the buffer for replays

        encoder (Encoder): the encoder
    """
    def __init__(self, entity_iden: int, replay_path=REPLAY_FOLDER, evaluation=False):
        self.entity_iden = entity_iden

        self.model = init_or_load_model(True)

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
        result = torch.zeros(OUTPUT_DIM, dtype=torch.float)
        self.teacher.classify(
            self.model,
            self.encoder.encode(game_state, None),
            result
        )
        return float(result[MOVE_LOOKUP[move]].item())

    def evaluate_all(self, game_state: GameState,
                     moves: typing.List[Move]) -> typing.Iterable[float]:
        """Evaluates all the moves for the given game state in a single pass"""
        result = torch.zeros(OUTPUT_DIM, dtype=torch.float)
        self.teacher.classify(
            self.model,
            self.encoder.encode(game_state, None),
            result
        )
        return result

    def learn(self, game_state: GameState, move: Move, new_state: GameState,
              reward_raw: float, reward_pred: float) -> None:
        if self.evaluation:
            #print(f'received reward {reward_raw} (pred: {reward_pred}) for move {move.name}')
            #best_move_val = self.evaluate_all(new_state, None).max().item()
            #print(f'  predicted value of next move: {best_move_val}')
            #print(f'  after adding {PRED_WEIGHT}*pred val to reward: ', end='')
            #print(f'{reward_raw + PRED_WEIGHT*best_move_val}')
            return
        player_id = 1 if self.entity_iden == game_state.player_1_iden else 2

        correct_bootstrapped_reward = (reward_raw + reward_pred)
        bootstrapped_reward = self.evaluate_all(game_state, MOVE_MAP).max().item()
        td_error = correct_bootstrapped_reward - bootstrapped_reward
        self.replay.add(replay_buffer.Experience(game_state, move, self.cutoff,
                                                 new_state, reward_raw, player_id, td_error,
                                                 self.encoder.encode(game_state, None),
                                                 self.encoder.encode(new_state, None)))

    def save(self) -> None:
        pass

class MyQBotController(qbot.QBotController):
    """Adds the pitch and supported moves for this to the qbot controller"""
    @classmethod
    def pitch(cls):
        return (
            'Deep2QBot',
            'Learns through offline replay of moves with a target network. '
            + 'Has 1 output feature per move.',
        )

    @classmethod
    def supported_moves(cls):
        return MOVE_MAP

def deep2(entity_iden: int, settings: str = None) -> 'Bot':  # noqa: F821
    """Creates a new deep2 bot"""
    if not settings:
        settings = SETTINGS_FILE
    if not os.path.exists(settings):
        raise FileNotFoundError(settings)

    with open(settings, 'r') as infile:
        settings = json.load(infile)

    return MyQBotController(
        entity_iden,
        Deep2QBot(entity_iden, settings['replay_path'],
                  (('eval' in settings) and settings['eval'])),
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
        target_model (Deep2Network): the network we use to evaluate states
        target_teacher (NetworkTeacher): the teacher we can use to evaluate states

        using_priority (bool): if True we are using priority sampling to fill, if False
            we are sampling uniformly. When using priority sampling, we fill recent
        recent (typing.List[float, float, Experience, float]):
                priority (float): the priority of the experience
                probability (float): the probability of this experience being selected
                exp (Experience): the experience that was selected

        __position (int): where we "are", faked
        marks (list[int]): the positions we have marked, faked for keeping track of epochs

        _buffer [torch.tensor[n, ENCODE_DIM]]: The buffer for inputs to the target network,
            cached to avoid unnecessary garbage. n is the largest batch size that we have
            seen so far.
        _outbuffer [torch.tensor[n, OUTPUT_DIM]]: The buffer used for outputs from
            the target network
        _outmask [torch.tensor[n, OUTPUT_DIM]]: A mask for the out buffer
    """
    def __init__(self, replay: replay_buffer.ReadableReplayBuffer, target_model: Deep2Network):
        super().__init__(len(replay), ENCODE_DIM, OUTPUT_DIM)
        self.replay = replay
        self.target_model = target_model
        self.target_teacher = FFTeacher()

        self.using_priority = False
        self.recent = None

        self._buffer = torch.zeros((1, ENCODE_DIM), dtype=torch.float)
        self._outbuffer = torch.zeros((1, OUTPUT_DIM), dtype=torch.float)
        self._outmask = torch.zeros((1, OUTPUT_DIM), dtype=torch.uint8)

        self.__position = 0
        self.marks = []

    def mark(self):
        self.replay.mark()
        self.marks.append(self.__position)

    def reset(self):
        self.replay.reset()
        self.__position = self.marks.pop()

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, val):
        raise NotImplementedError

    @property
    def remaining_in_epoch(self):
        return len(self.replay) - self.__position

    def __next__(self) -> pwl.PointWithLabel:
        raise NotImplementedError('no reasonable way to do this since '
                                  + 'we need to encode the move in the label')

    def _ensure_buffers(self, batch_size: int):
        if self._buffer.shape[0] >= batch_size:
            return

        self._buffer = torch.zeros((batch_size, ENCODE_DIM), dtype=torch.float)
        self._outbuffer = torch.zeros((batch_size, OUTPUT_DIM), dtype=torch.float)
        self._outmask = torch.zeros((batch_size, OUTPUT_DIM), dtype=torch.uint8)

    def fill(self, points: torch.tensor, labels: torch.tensor) -> None:
        batch_size = points.shape[0]
        self.__position = (self.__position + batch_size) % len(self.replay)
        self._ensure_buffers(batch_size)

        labels[:] = INVALID_REWARD
        self._outmask[:] = 0

        if not self.using_priority:
            exps = self.replay.sample(batch_size)
        else:
            exps = []
            self.recent = []
            self.using_priority = False

            for _ in range(batch_size):
                prio, prob, exp = self.replay.pop()
                exps.append(exp)
                self.recent.append([prio, prob, exp])

        for i, exp in enumerate(exps):
            exp: replay_buffer.Experience
            points[i, :] = torch.from_numpy(exp.encoded_state)
            self._buffer[i, :] = torch.from_numpy(exp.new_encoded_state)
            labels[i, MOVE_LOOKUP[exp.action]] = exp.reward_rec
            self._outmask[i, MOVE_LOOKUP[exp.action]] = 1
        self.target_teacher.classify_many(self.target_model, self._buffer[:batch_size],
                                          self._outbuffer[:batch_size])
        labels[self._outmask] += self._outbuffer[:batch_size].max(dim=1)[0] * PRED_WEIGHT


    def _fill(self, points, labels):
        raise NotImplementedError

    def _position(self, pos):
        raise NotImplementedError

class MyCrit:
    """The criterion we use for evaluating, which can have weights injected into it. This
    is mean square error with a regularizing factor

    Attributes:
        regul_factor (float): how much the regularization term effects the loss
        sample_weights (torch.tensor[batch_size], optional): if specified, the next call
            to this criterion will use the given sample weights as an element-wise product
            with the loss.
    """
    def __init__(self, regul_factor: float):
        self.regul_factor = regul_factor
        self.sample_weights = None

    def __call__(self, pred: torch.tensor, truth: torch.tensor):
        known_val = truth != INVALID_REWARD

        sq_diffs = (pred[known_val] - truth[known_val]) ** 2
        if self.sample_weights is not None:
            sq_diffs = sq_diffs * self.sample_weights
            self.sample_weights = None
        loss = sq_diffs.mean()

        loss += self.regul_factor * (pred ** 2).mean() # regularizer
        return loss

class UsePrioritySampling:
    """Should be injected into the END of the trainer"""
    def pre_loop(self, context: tnr.GenericTrainingContext) -> None:
        """Tells the PWL to use priority sampling on the next fill"""
        context.train_pwl.using_priority = True

    def pre_train(self, context: tnr.GenericTrainingContext) -> None:
        """Tells the FFTeacher to store the teach result"""
        context.teacher.store_teach_result = True

class UsePrioritySampling2:
    """Should be injected into the START of the trainer"""
    def __init__(self, crit: MyCrit, beta: float):
        self.criterion = crit
        self.beta = beta
        self._buffer = None

    def _ensure_buffers(self, context: tnr.GenericTrainingContext) -> None:
        batch_size = context.batch_size
        if self._buffer is None or self._buffer.shape[0] != batch_size:
            self._buffer = torch.zeros(batch_size, dtype=torch.float)

    def post_points(self, context: tnr.GenericTrainingContext) -> None:
        """Applies the importance sampling weights to the criterion"""
        self._ensure_buffers(context)
        for i in range(context.batch_size):
            self._buffer[i] = context.train_pwl.recent[i][1]
        self._buffer = ((1 / len(context.train_pwl.replay)) * (1 / self._buffer)) ** self.beta
        self._buffer /= self._buffer.max()
        self.criterion.sample_weights = self._buffer

    def post_train(self, context: tnr.GenericTrainingContext, loss: float) -> None:
        """Inserts the trained experiences back into the replay buffer"""
        net_outs = context.teacher.teach_result
        context.teacher.store_teach_result = False

        for i, (_, _, exp) in enumerate(context.train_pwl.recent):
            net_pred = net_outs[i][MOVE_LOOKUP[exp.action]].item()
            cor_pred = context.labels[i]
            cor_pred = cor_pred[cor_pred != -2].item()
            td_error = cor_pred - net_pred
            exp.last_td_error = td_error
            context.train_pwl.replay.add(exp)


def _disp_acts(net, mpwl):
    num_points = 32
    sample_points = torch.zeros((num_points, ENCODE_DIM), dtype=torch.float)
    sample_labels = torch.zeros((num_points, OUTPUT_DIM), dtype=torch.float)

    mpwl.mark()
    mpwl.fill(sample_points, sample_labels)
    mpwl.reset()

    hid_acts = []

    def on_hidacts(acts_info: FFHiddenActivations):
        hid_acts.append(acts_info.hidden_acts.detach())

    net(sample_points, on_hidacts)

    for i, lyr in enumerate(hid_acts):
        print(f'Layer {i}: {lyr}')

def offline_learning():
    """Loads the replay buffer and trains on it."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluates the deep1 bot by launching a server and connecting it')
    parser.add_argument('regul_factor', type=float,
                        help='The weight of regularization, should be proportional to '
                        + 'teacher force amount')
    parser.add_argument('alpha', type=float,
                        help='How much prioritization is used, with alpha=0 corresponding '
                        + 'to uniform replay. As alpha tends to infinity, this tends towards '
                        + 'completely deterministic. Typically 0<alpha<1')
    parser.add_argument('beta', type=float,
                        help='Importance sampling weights for learning rates to compensate '
                        + 'for prioritized replay. 0<=beta<=1, where beta=1 fully compensates'
                        + ' and beta=0 does not compensate at all')
    args = parser.parse_args()

    perf_file = os.path.join(SAVEDIR, 'offline_learning_perf.log')
    perf = perf_stats.LoggingPerfStats('deep2 offline learning', perf_file)

    replay = replay_buffer.MemoryPrioritizedReplayBuffer(REPLAY_FOLDER, alpha=args.alpha)
    try:
        print(f'Replaying {len(replay)} experiences with prioritized replay ('
              + f'regul_factor={args.regul_factor}, alpha={args.alpha}, beta={args.beta})')
        network = init_or_load_model()
        teacher = FFTeacher()

        train_pwl = MyPWL(replay, init_or_load_model(True))
        test_pwl = train_pwl

        def update_target(ctx: tnr.GenericTrainingContext, hint: str):
            ctx.logger.info('swapping target network, hint=%s', hint)
            network.save(MODELFILE, exist_ok=True)

            new_target = Deep2Network.load(MODELFILE)
            new_target.start_stat_tracking()
            for _ in range(3):
                train_pwl.mark()
                for _ in range(0, 1024, ctx.batch_size):
                    train_pwl.fill(ctx.points, ctx.labels)
                    teacher.classify_many(new_target, ctx.points, ctx.labels)
                new_target.stat_tracking_to_norm()
                train_pwl.reset()
            new_target.stop_stat_tracking()
            new_target.save(EVAL_MODELFILE, exist_ok=True)

            train_pwl.target_model = new_target

        trainer = tnr.GenericTrainer(
            train_pwl=train_pwl,
            test_pwl=test_pwl,
            teacher=teacher,
            batch_size=32,
            learning_rate=0.00025,
            optimizer=torch.optim.Adam(
                [p for p in network.parameters() if p.requires_grad], lr=0.001),
            criterion=MyCrit(args.regul_factor)
        )
        (trainer
         .reg(UsePrioritySampling2(trainer.criterion, args.beta))
         .reg(tnr.EpochsTracker())
         .reg(tnr.EpochsStopper(3))
         .reg(tnr.InfOrNANDetecter())
         .reg(tnr.InfOrNANStopper())
         .reg(tnr.DecayTracker())
         .reg(tnr.DecayStopper(1))
         .reg(tnr.OnEpochCaller.create_every(update_target, skip=CUTOFF)) # smaller cutoffs require more bootstrapping pylint: disable=line-too-long
         .reg(tnr.DecayOnPlateau())
         .reg(UsePrioritySampling())
        )
        res = trainer.train(network, target_dtype=torch.float32,
                            point_dtype=torch.float32,
                            target_style='hot',
                            perf=perf)
        if res['inf_or_nan']:
            print('training failed! inf or nan!')
    finally:
        flattened = replay.flatten()
        replay.close()

        print('resaving replay')
        filetools.deldir(REPLAY_FOLDER)
        replay = replay_buffer.FileWritableReplayBuffer(REPLAY_FOLDER)
        for exp in flattened:
            replay.add(exp)
        replay.close()
        print('finished resaving replay')

if __name__ == '__main__':
    offline_learning()
