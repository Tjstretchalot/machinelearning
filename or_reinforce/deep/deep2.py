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
import scipy.io
import datetime

from shared.models.ff import FeedforwardNetwork, FFTeacher, FFHiddenActivations
from shared.layers.norm import EvaluatingAbsoluteNormLayer, LearningAbsoluteNormLayer
from shared.layers.affine import AffineLayer
import shared.nonlinearities as snonlins
import shared.trainer as tnr
import shared.perf_stats as perf_stats
import shared.typeutils as tus
import shared.cp_utils as cp_utils
import shared.filetools as filetools
import shared.measures.utils as mutils

import or_reinforce.utils.qbot as qbot
import or_reinforce.utils.rewarders as rewarders
import or_reinforce.utils.encoders as encoders
import or_reinforce.utils.general as gen
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
        (
            encoders.LocationEncoder(entity_iden),
            encoders.StaircaseLocationEncoder(entity_iden),
        )
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
        inp_layer (StackableLayer): the input layer
        extract_layers (list[StackableLayer]): the feature extraction layers
        out_layer (torch.nn.Linear): the output layer
        out_nonlin_nm (str): the name for the output nonlinearity (see shared.nonlinearites)
        out_nonlin (typing.Callable): the output nonlinearity
    """
    def __init__(self, inp_layer: StackableLayer,
                 extract_layers: typing.List[StackableLayer],
                 out_layer: torch.nn.Linear,
                 out_nonlin_nm: str):
        super().__init__(ENCODE_DIM, OUTPUT_DIM, 1 + len(extract_layers))
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
            StackableLayer.create(ENCODE_DIM, HIDDEN_DIM),
            [StackableLayer.create(HIDDEN_DIM, HIDDEN_DIM) for i in range(5)],
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False),
            'tanh'
        )

    def identity_eval(self) -> None:
        """Swaps the normalization layers with identity layers. Required when you dont have any
        data to fetch statistics on but you can't use batch norms"""
        self.inp_layer.identity_eval()
        for lyr in self.extract_layers:
            lyr.identity_eval()

    def start_stat_tracking(self) -> None:
        """Begins switching this network to evaluation mode. In evaluation mode there are no
        batch norm layers - they are instead replaced with absolute normalization layers. This
        inserts learning norm layers prior to the batch norm layers, which can be converted
        to absolute layers after this network has been shown some samples"""
        self.inp_layer.start_stat_tracking()
        for lyr in self.extract_layers:
            lyr.start_stat_tracking()

    def stat_tracking_to_norm(self):
        """Swaps the norm layers with the current learned norm layers and resets the
        learning layers"""
        self.inp_layer.stat_tracking_to_norm()
        for lyr in self.extract_layers:
            lyr.stat_tracking_to_norm()

    def stop_stat_tracking(self):
        """Stops tracking statistics between layers without changing the current norm"""
        self.inp_layer.stop_stat_tracking()
        for lyr in self.extract_layers:
            lyr.stop_stat_tracking()

    def to_train(self):
        """Switches the norm layers to batch norm layers which are suitable for training"""
        self.inp_layer.to_train()
        for lyr in self.extract_layers:
            lyr.to_train()

    def forward(self, inps: torch.tensor, acts_cb: typing.Callable = _noop) -> torch.tensor: # pylint: disable=arguments-differ
        acts_cb(FFHiddenActivations(layer=0, hidden_acts=inps))
        acts = self.inp_layer(inps)
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

        npres = dict()
        primres = {'num_extract': len(self.extract_layers)}

        lyr_npres, lyr_primres = self.inp_layer.to_raw_numpy()
        for k, v in lyr_npres.items():
            npres[f'inp_{k}'] = v
        primres['inp'] = lyr_primres

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
            inp = StackableLayer.from_raw_numpy(npres, prims['inp'], 'inp_')

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

        return cls(inp, extract_layers, out_layer, out_nonlin_nm)

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
        encoders_by_id (dict[int, Encoder]): the encoders by player id

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
        self.encoders_by_id = dict()

        self._buffer = torch.zeros((1, ENCODE_DIM), dtype=torch.float)
        self._outbuffer = torch.zeros((1, OUTPUT_DIM), dtype=torch.float)
        self._outmask = torch.zeros((1, OUTPUT_DIM), dtype=torch.uint8)

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
        self._ensure_buffers(batch_size)

        labels[:] = INVALID_REWARD
        self._outmask[:] = 0

        exps = self.replay.sample(batch_size)
        for i, exp in enumerate(exps):
            exp: replay_buffer.Experience
            ent_id = exp.state.player_1_iden if exp.player_id == 1 else exp.state.player_2_iden
            if ent_id not in self.encoders_by_id:
                self.encoders_by_id[ent_id] = init_encoder(ent_id)

            enc: encoders.FlatEncoder = self.encoders_by_id[ent_id]
            enc.encode(exp.state, None, out=points[i])
            enc.encode(exp.new_state, None, out=self._buffer[i])
            labels[i, MOVE_LOOKUP[exp.action]] = exp.reward_rec
            self._outmask[i, MOVE_LOOKUP[exp.action]] = 1

        self.target_teacher.classify_many(self.target_model, self._buffer[:batch_size],
                                          self._outbuffer[:batch_size])
        labels[self._outmask] += self._outbuffer[:batch_size].max(dim=1)[0] * PRED_WEIGHT

    def _fill(self, points, labels):
        raise NotImplementedError

    def _position(self, pos):
        raise NotImplementedError

def _crit(pred: torch.tensor, truth: torch.tensor):
    known_val = truth != INVALID_REWARD

    loss = torch.functional.F.smooth_l1_loss(
        pred[known_val].unsqueeze(1), truth[known_val].unsqueeze(1)
    )

    loss += 0.1 * (pred ** 2).sum() # regularizer
    return loss

def offline_learning():
    """Loads the replay buffer and trains on it."""
    perf_file = os.path.join(SAVEDIR, 'offline_learning_perf.log')
    perf = perf_stats.LoggingPerfStats('deep2 offline learning', perf_file)

    replay = replay_buffer.FileReadableReplayBuffer(REPLAY_FOLDER, perf=perf)
    try:
        print(f'loaded {len(replay)} experiences for replay...')
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
            learning_rate=0.001,
            optimizer=torch.optim.Adam(
                [p for p in network.parameters() if p.requires_grad], lr=0.001),
            criterion=_crit
        )
        (trainer
         .reg(tnr.EpochsTracker())
         .reg(tnr.EpochsStopper(100))
         .reg(tnr.InfOrNANDetecter())
         .reg(tnr.InfOrNANStopper())
         .reg(tnr.DecayTracker())
         .reg(tnr.DecayStopper(1))
         .reg(tnr.OnEpochCaller.create_every(update_target, skip=CUTOFF)) # smaller cutoffs require more bootstrapping pylint: disable=line-too-long
         .reg(tnr.DecayOnPlateau())
        )
        res = trainer.train(network, target_dtype=torch.float32,
                            point_dtype=torch.float32,
                            target_style='hot',
                            perf=perf)
        if res['inf_or_nan']:
            print('training failed! inf or nan!')
    finally:
        replay.close()

if __name__ == '__main__':
    offline_learning()
