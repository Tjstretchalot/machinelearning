"""This is used for the replay buffer of experiences. The replay buffer is well-described
in this article: https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4
"""
import typing
import os
import json
import struct
import io
import random
from collections import deque
import numpy as np

import shared.typeutils as tus
import shared.perf_stats as perf_stats
from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
import optimax_rogue.networking.serializer as ser

class Experience(ser.Serializable):
    """Describes an experience which can be put into or retrieved from the replay buffer.

    Attributes:
        state (GameState): the state of the game
        action (Move): the action that was taken from the state
        delay (int): the number of ticks we simulated
        new_state (GameState): the new state that we were in after
        reward_rec (float): the reward that we actually received in the delay ticks
        player_id (int): either 1 or 2 for the player the bot is controlling
        last_td_error (float, optional): when last checked, what was the temporal difference between
            the value we expected from the state-action and what we saw
        encoded_state (np.ndarray, float32, flat): the encoded initial state
        new_encoded_state (np.ndarray, float32, flat): the encoded final state
    """
    def __init__(self, state: GameState, action: Move, delay: int, new_state: GameState,
                 reward_rec: float, player_id: int, last_td_error: typing.Optional[float],
                 encoded_state: np.ndarray, new_encoded_state: np.ndarray):
        tus.check(state=(state, (GameState, bytes)), action=(action, Move), delay=(delay, int),
                  new_state=(new_state, (GameState, bytes)),
                  reward_rec=(reward_rec, float), player_id=(player_id, int),
                  last_td_error=(last_td_error, (type(None), float)),
                  encoded_state=(encoded_state, np.ndarray),
                  new_encoded_state=(new_encoded_state, np.ndarray))
        if encoded_state.dtype != np.dtype('float32'):
            raise ValueError(f'expected encoded state dtype is float32, got {encoded_state.dtype}')
        if new_encoded_state.dtype != np.dtype('float32'):
            raise ValueError('expected new encoded state dtypei s float32, '
                             + f'got {new_encoded_state.dtype}')
        if len(encoded_state.shape) != 1:
            raise ValueError(f'expected encoded state is flat but has shape {encoded_state.shape}')
        if len(new_encoded_state.shape) != 1:
            raise ValueError('expected new encoded state is flat, but has shape '
                             + str(new_encoded_state.shape))

        self._state = state
        self.action = action
        self.delay = delay
        self._new_state = new_state
        self.reward_rec = reward_rec
        self.player_id = player_id
        self.last_td_error = last_td_error
        self.encoded_state = encoded_state
        self.new_encoded_state = new_encoded_state

    @property
    def state(self) -> GameState:
        """Returns the game state we started in. Lazily decoded"""
        if isinstance(self._state, GameState):
            return self._state
        self._state = GameState.from_prims(self._state)
        return self._state

    @property
    def new_state(self) -> GameState:
        """Returns the game state we ended in, lazily decoded"""
        if isinstance(self._new_state, GameState):
            return self._new_state
        self._new_state = GameState.from_prims(self._new_state)
        return self._new_state

    def has_custom_serializer(self):
        return True

    def to_prims(self) -> bytes:
        arr = io.BytesIO()
        serd_state = (
            self._state.to_prims()
            if isinstance(self._state, GameState)
            else self._state)
        arr.write(len(serd_state).to_bytes(4, 'big', signed=False))
        arr.write(serd_state)
        arr.write(int(self.action).to_bytes(4, 'big', signed=False))
        arr.write(self.delay.to_bytes(4, 'big', signed=False))
        serd_state = (
            self._new_state.to_prims()
            if isinstance(self._new_state, GameState)
            else self._new_state)
        arr.write(len(serd_state).to_bytes(4, 'big', signed=False))
        arr.write(serd_state)
        arr.write(struct.pack('>f', self.reward_rec))
        arr.write(self.player_id.to_bytes(4, 'big', signed=False))
        if self.last_td_error:
            arr.write((1).to_bytes(1, 'big', signed=False))
            arr.write(struct.pack('>f', self.last_td_error))
        else:
            arr.write((0).to_bytes(1, 'big', signed=False))

        serd_state = self.encoded_state.tobytes()
        arr.write(len(serd_state).to_bytes(4, 'big', signed=False))
        arr.write(serd_state)
        serd_state = self.new_encoded_state.tobytes()
        arr.write(len(serd_state).to_bytes(4, 'big', signed=False))
        arr.write(serd_state)
        return arr.getvalue()

    @classmethod
    def from_prims(cls, prims: bytes) -> 'Experience':
        arr = io.BytesIO(prims)
        arr.seek(0, 0)
        state_len = int.from_bytes(arr.read(4), 'big', signed=False)
        state = arr.read(state_len)
        action = Move(int.from_bytes(arr.read(4), 'big', signed=False))
        delay = int.from_bytes(arr.read(4), 'big', signed=False)
        state_len = int.from_bytes(arr.read(4), 'big', signed=False)
        new_state = arr.read(state_len)
        reward = struct.unpack('>f', arr.read(4))[0]
        player_id = int.from_bytes(arr.read(4), 'big', signed=False)
        have_td = int.from_bytes(arr.read(1), 'big', signed=False) == 1
        if have_td:
            last_td_error = struct.unpack('>f', arr.read(4))[0]
        else:
            last_td_error = None

        state_len = int.from_bytes(arr.read(4), 'big', signed=False)
        encoded_state = np.frombuffer(arr.read(state_len), dtype='float32')

        state_len = int.from_bytes(arr.read(4), 'big', signed=False)
        encoded_new_state = np.frombuffer(arr.read(state_len), dtype='float32')
        return Experience(state, action, delay, new_state, reward, player_id, last_td_error,
                          encoded_state, encoded_new_state)

    def __eq__(self, other: 'Experience') -> bool:
        if not isinstance(other, Experience):
            return False
        if isinstance(self._state, bytes) and isinstance(other._state, bytes): # pylint: disable=protected-access, line-too-long
            if self._state != other._state: # pylint: disable=protected-access
                return False
        elif self.state != other.state:
            return False
        if self.action != other.action:
            return False
        if self.delay != other.delay:
            return False
        if isinstance(self._state, bytes) and isinstance(other._state, bytes): # pylint: disable=protected-access, line-too-long
            if self._state != other._state: # pylint: disable=protected-access
                return False
        elif self.new_state != other.new_state:
            return False
        if abs(self.reward_rec - other.reward_rec) > 1e-6:
            return False
        if self.player_id != other.player_id:
            return False
        if self.last_td_error:
            if not other.last_td_error:
                return False
            if abs(self.last_td_error - other.last_td_error) > 1e-6:
                return False
        elif other.last_td_error:
            return False
        return True

    def __repr__(self) -> str:
        return (f'Experience [state={repr(self.state)}, action={repr(self.action)}, '
                + f'new_state={repr(self.new_state)}, '
                + f'reward_rec={repr(self.reward_rec)}, player_id={self.player_id}, '
                + f'last_td_error={repr(self.last_td_error)}]')

ser.register(Experience)

class WritableReplayBuffer:
    """The generic interface for a replay buffer that you can write to"""
    def __len__(self) -> int:
        """Returns the number of elements in the buffer"""
        raise NotImplementedError

    def add(self, experience: Experience) -> None:
        """Adds the given experience to the buffer"""
        raise NotImplementedError

    def close(self) -> None:
        """Closes this write buffer"""
        raise NotImplementedError

class ReadableReplayBuffer:
    """The generic interface for a replay buffer that you can read from"""
    def __len__(self) -> int:
        """Returns the number of elements in the buffer. This will not have an effect on sampling
        but might dictate how much use you can get out of this buffer."""
        raise NotImplementedError

    def mark(self) -> None:
        """Marks this buffer such that all the experiences retrieved from sample()
        will be retrieved in the same order once reset() is called
        """
        raise NotImplementedError

    def reset(self):
        """Resets the buffer to the last marked position"""
        raise NotImplementedError

    def sample(self, num: int) -> typing.Iterable[Experience]:
        """Fetches the specified number of experiences from the replay buffer."""
        raise NotImplementedError

    def close(self) -> None:
        """Closes this read buffer"""
        raise NotImplementedError

EXPERIENCES_FILE = 'experiences.dat'
META_FILE = 'meta.json'
class FileWritableReplayBuffer(WritableReplayBuffer):
    """An implementation of WritableReplayBuffer that stores experiences in a particular
    file.

    Attributes:
        _len (int): the number of elements in this buffer
        _flen (int): the file length for integrity checking
        _largest_state_nbytes (int): the number of bytes in the largest state
        dirname (str): the path to the folder where we can store stuff
        handle (file): the handle to the file that we are writing to
    """
    def __init__(self, dirname: str, exist_ok: bool = False):
        tus.check(dirname=(dirname, str))

        self.dirname = dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            self.handle = open(os.path.join(self.dirname, EXPERIENCES_FILE), 'wb')
            self._len = 0
            self._flen = 0
            self._largest_state_nbytes = 0
            self._write_meta()
        elif not os.path.isdir(dirname):
            raise ValueError(f'dirname must be a path to a directory, but {dirname} '
                             + f'exists and is not a directory')
        else:
            with open(os.path.join(self.dirname, META_FILE), 'r') as infile:
                meta = json.load(infile)
                self._len = meta['length']
                self._flen = meta['file_length']
                self._largest_state_nbytes = meta['largest_state_nbytes']
            expfile = os.path.join(self.dirname, EXPERIENCES_FILE)
            exp_len = os.path.getsize(expfile)
            if self._flen != exp_len:
                import warnings
                warnings.warn(f'experiences file corrupted - written up to {exp_len}, '
                              + f'valid up to {self._flen}. will discard bad data')
                if exp_len < self._flen:
                    raise ValueError(f'cannot recover experience file smaller than '
                                     + f'expected (got {exp_len}, expected {self._flen})')

                counter = 1
                cp_loc = os.path.join(self.dirname,
                                      EXPERIENCES_FILE + '.' + str(counter) + '.corrupted')
                while os.path.exists(cp_loc):
                    counter += 1
                    cp_loc = os.path.join(self.dirname,
                                          EXPERIENCES_FILE + '.' + str(counter) + '.corrupted')

                os.rename(expfile, cp_loc)
                with open(cp_loc, 'rb') as infile:
                    with open(expfile, 'wb') as outfile:
                        bytes_rem = self._flen
                        while bytes_rem >= 4096:
                            outfile.write(infile.read(4096))
                            bytes_rem = bytes_rem - 4096
                        if bytes_rem > 0:
                            outfile.write(infile.read(bytes_rem))

                self._write_meta()

            self.handle = open(expfile, 'ab')

    def __del__(self):
        self.close()

    def _write_meta(self):
        with open(os.path.join(self.dirname, META_FILE), 'w') as outfile:
            json.dump({'length': self._len, 'file_length': self._flen,
                       'largest_state_nbytes': self._largest_state_nbytes}, outfile)

    def __len__(self):
        return self._len

    def add(self, experience: Experience) -> None:
        if not self.handle:
            raise ValueError('handle closed already')

        to_w = experience.to_prims()
        self.handle.write(len(to_w).to_bytes(4, 'big', signed=False))
        nwritten = self.handle.write(to_w)
        while nwritten < len(to_w):
            nwritten += self.handle.write(to_w[nwritten:])
        self.handle.flush()
        self._len += 1
        self._flen += len(to_w) + 4
        self._largest_state_nbytes = max(self._largest_state_nbytes, len(to_w))
        self._write_meta()

    def close(self) -> None:
        if hasattr(self, 'handle') and self.handle:
            self.handle.close()
            self.handle = None

class FileMark:
    """Describes a mark in a FileReadableReplayBuffer. Not meant to be used directly.

    Attributes:
        shuffle_counter (int): the index to the shuffle
        file_loc (int): the location within the file that the experience that was marked
            starts at
    """
    def __init__(self, shuffle_counter: int, file_loc: int) -> None:
        self.shuffle_counter = shuffle_counter
        self.file_loc = file_loc

class FileReadableReplayBuffer(ReadableReplayBuffer):
    """A readable replay buffer that takes in the folder written to by a
    FileWritableReplayBuffer. For shuffling a copy is created that has the experiences
    in a random order. The memory requirements of this class scale only with the
    number of marks.

    Attributes:
        _len (int): the length of the entire buffer
        _padded_size (int): the number of bytes we pad each state to for faster reading
        dirname (str): the folder that
        handle (file): the handle to the shuffled experiences file we are currently viewing
        shuffle_counter (int): the current shuffle we are at. Shuffling copies the file, and
            the shuffled file is deleted once we are no longer using it and there are no marks
            to it.
        marks (list[FileMark]): the marks that have been placed, where lower indices are older.

        _block (bytearray([self.padded_size])): used for reading
        _blockmv (memoryview(_block)): memory view of block

        perf (PerfStats): the performance tracker
    """

    def __init__(self, dirname: str, perf: typing.Optional[perf_stats.PerfStats] = None) -> None:
        tus.check(dirname=(dirname, str), perf=(perf, (type(None), perf_stats.PerfStats)))

        if not perf:
            perf = perf_stats.NoopPerfStats()
        if not os.path.exists(dirname):
            raise FileNotFoundError(dirname)
        if not os.path.isdir(dirname):
            raise ValueError(f'{dirname} is not a folder')

        self.dirname = dirname
        self.perf: perf_stats.PerfStats = perf

        with open(os.path.join(dirname, META_FILE), 'r') as infile:
            meta = json.load(infile)
            self._len = meta['length']
            largest_nbytes = meta['largest_state_nbytes']
            self.padded_size = (
                (largest_nbytes + 4 + 4095) // 4096) * 4096 # need 4 bytes for length

        self._shuffle(1)
        self.handle = open(self._shuffle_path(1), 'rb', buffering=0)
        self.shuffle_counter = 1
        self.marks = []
        self._block = bytearray(self.padded_size)
        self._blockmv = memoryview(self._block)

    def _shuffle(self, counter: int) -> None:
        inpath = os.path.join(self.dirname, EXPERIENCES_FILE)
        outpath = self._shuffle_path(counter)

        if not os.path.exists(inpath):
            raise FileNotFoundError(inpath)
        if os.path.exists(outpath):
            raise FileExistsError(outpath)

        with open(inpath, 'rb') as infile:
            with open(outpath, 'xb+', buffering=0) as outfile:
                block = bytearray(self.padded_size)
                block2 = bytearray(self.padded_size)
                blockmv = memoryview(block)
                block2mv = memoryview(block2)
                lastlen = 0

                for i in range(self._len):
                    if outfile.tell() != i * self.padded_size:
                        raise ValueError(f'should not get here; outfile loc = {outfile.tell()}, '
                                         + f'expected = {i * self.padded_size}')

                    inlen = int.from_bytes(infile.read(4), 'big', signed=False)
                    block[0:4] = inlen.to_bytes(4, 'big', signed=False)
                    bytes_read = infile.readinto(blockmv[4:4 + inlen])
                    _ctr = 0
                    while bytes_read < inlen:
                        bytes_read += infile.readinto(blockmv[4 + bytes_read:4 + inlen])
                        _ctr += 1
                        if _ctr > 100:
                            raise ValueError(f'infinite loop detected: bytes_read={bytes_read}, '
                                             + f'expected = {inlen}')
                    if lastlen > inlen:
                        blockmv[4 + inlen:4 + lastlen] = bytes(lastlen - inlen)
                    lastlen = inlen

                    j = random.randint(0, i)
                    if i != j:
                        # goal: at position i (current file location), write the value at
                        # position j (old file location)
                        ipos = outfile.tell() # store pos at i
                        outfile.seek(j * self.padded_size, 0) # seek to j
                        bytesread = outfile.readinto(block2) # read from j
                        _ctr = 0
                        while bytesread < self.padded_size: # really get everything
                            bytesread += outfile.readinto(block2mv[bytesread:])
                            _ctr += 1
                            if _ctr > 100:
                                raise ValueError(f'infinite loop detected; bytesread = {bytesread}'
                                                 + f', expected = {self.padded_size}')
                        outfile.seek(ipos) # seek to i
                        outfile.write(block2) # write from j
                        outfile.seek(j * self.padded_size, 0) # seek to j
                        outfile.write(block) # write source i
                        outfile.seek((i + 1) * self.padded_size, 0) # seek to end
                    else:
                        # goal: at position i, write source i
                        outfile.write(block)

    def _shuffle_path(self, counter: int) -> str:
        return os.path.join(self.dirname, f'experiences-shuffle-{counter}.dat')

    @property
    def position(self):
        """Gets were we are in the current epoch. The same position will not generally correspond
        with the same experience due to shuffling"""
        return self.handle.tell() // self.padded_size

    @property
    def remaining_in_epoch(self):
        """How many more experiences there are in the buffer before we shuffle"""
        return self._len - self.position

    def mark(self):
        self.marks.append(FileMark(shuffle_counter=self.shuffle_counter,
                                   file_loc=self.handle.tell()))

    def reset(self):
        to_mark = self.marks.pop()
        self.shuffle_counter = to_mark.shuffle_counter
        if self.shuffle_counter != to_mark.shuffle_counter:
            self.handle.close()
            self.handle = open(self._shuffle_path(to_mark.shuffle_counter), 'rb', buffering=0)
        self.handle.seek(to_mark.file_loc, 0)

    def sample(self, num: int) -> typing.Iterable[Experience]:
        result = []

        for _ in range(num):
            result.append(next(self))

        return result

    def __len__(self) -> int:
        return self._len

    def __next__(self) -> Experience:
        self.perf.enter('REPLAY_BUFFER_NEXT')
        self.perf.enter('READ_BLOCK')
        bytes_read = self.handle.readinto(self._block)
        if bytes_read == 0:
            raise ValueError('am at end of file but ought not be')

        while bytes_read < self.padded_size:
            amt = self.handle.readinto(self._blockmv[bytes_read:])
            if amt == 0:
                raise ValueError('am at end of file but ought not be')
            bytes_read += amt
        self.perf.exit_then_enter('DECODE_BLOCK')
        reqlen = int.from_bytes(self._blockmv[:4], 'big', signed=False)
        exp = Experience.from_prims(self._blockmv[4:4 + reqlen])
        self.perf.exit()

        if self.handle.tell() >= self._len * self.padded_size:
            self.perf.enter('SHUFFLE')
            old_path = self._shuffle_path(self.shuffle_counter)
            next_path = self._shuffle_path(self.shuffle_counter + 1)
            if not os.path.exists(next_path):
                self._shuffle(self.shuffle_counter + 1)
            self.handle.close()
            self.handle = open(next_path, 'rb', buffering=0)

            can_del_old = True
            for mark in self.marks:
                if mark.shuffle_counter <= self.shuffle_counter:
                    can_del_old = False
                    break

            if can_del_old:
                os.remove(old_path)

            self.shuffle_counter += 1
            self.perf.exit()
        self.perf.exit()
        return exp

    def close(self) -> None:
        if not self.handle:
            return

        self.handle.close()
        self.handle = None

        earliest_file = self.shuffle_counter
        if self.marks:
            import warnings
            warnings.warn('replay buffer leaking marks', UserWarning)

        for mark in self.marks:
            earliest_file = min(earliest_file, mark.shuffle_counter)

        for i in range(earliest_file, self.shuffle_counter + 1):
            os.remove(self._shuffle_path(i))

BALANCE_SIZE = 1024
BALANCE_SIZE_POWS = [BALANCE_SIZE ** i for i in range(4)]
class PrioritizedExperienceNode:
    """A tree node in the prioritized replay buffer

    Attributes:
        priority (float): if this is a leaf node, this is the priority that
            the child should be picked. If this is not a leaf node, this is the
            sum of the priorities that the children should be picked
        alpha_norm_priority (float): the alpha-norm (where alpha is some constant
            float value) of the priorities inside this node.
        alpha (float): the alpha value used by this node. changing requires updating
            the alpha norm.
        children (list[PrioritizedExperienceNode], optional)
        child (Experience, optional)

        _len (int): how many experiences are stored in this node
        _flat (bool): true if all our children are leafs, false otherwise
    """
    def __init__(self, alpha: float, child: typing.Optional[Experience] = None,
                 priority: float = 0):
        self.alpha = alpha
        self.alpha_norm_priority = priority ** alpha
        self.priority = priority
        self.children = None
        self.child = child
        self._flat = True

        if child is None:
            self._len = 0
        else:
            self._len = 1

    def add(self, experience: Experience, priority: float):
        """Adds the given experience to this node"""
        if self._len == 0:
            self.child = experience
            self.priority = priority
            self.alpha_norm_priority = priority ** self.alpha
            self._len = 1
            return

        if self._len == 1:
            self.children = [
                PrioritizedExperienceNode(self.alpha, child=self.child, priority=self.priority),
                PrioritizedExperienceNode(self.alpha, child=experience, priority=priority)
            ]
            self.priority += priority
            self.alpha_norm_priority += self.children[1].alpha_norm_priority
            self._len = 2
            return

        if len(self.children) < BALANCE_SIZE:
            self.children.append(
                PrioritizedExperienceNode(self.alpha, child=experience, priority=priority))
            self.priority += priority
            self.alpha_norm_priority += self.children[-1].alpha_norm_priority
            self._len += 1
            return

        self._flat = False
        cutoff = 2
        while BALANCE_SIZE_POWS[cutoff] < self._len:
            cutoff += 1
        cutoff_child = BALANCE_SIZE_POWS[cutoff - 1]
        cutoff = BALANCE_SIZE_POWS[cutoff]

        sind = random.randrange(0, BALANCE_SIZE)
        for i in range(BALANCE_SIZE):
            child = self.children[i]
            if len(child) < cutoff_child:
                child.add(experience, priority)
                break
            sind += 1
            if sind >= BALANCE_SIZE:
                sind -= BALANCE_SIZE
        self.priority += priority
        self._len += 1
        self.alpha_norm_priority += priority ** self.alpha

    def flatten(self) -> typing.List[Experience]:
        """Converts this node into a list of experiences"""
        if self._len == 1:
            return [self.child]

        if self._flat:
            return [child.child for child in self.children]

        res = []
        for child in self.children:
            res.extend(child.flatten())
        return res

    def fetch_uniform(self) -> Experience:
        """Fetches an experience uniformly randomly from this node, without respect to
        priorities"""
        return self.fetch_at(random.randrange(0, self._len))

    def fetch_at(self, target_index: int) -> Experience:
        """Fetches the experience at the given index in this node"""
        if not 0 <= target_index < self._len:
            raise IndexError(f'index outside of range (expected in [0, {self._len}), '
                             + f'got {target_index})')
        if self._len == 1:
            return self.child
        if self._flat:
            return self.children[target_index].child
        cur_index = 0
        rol_sum = 0
        while rol_sum + len(self.children[cur_index]) <= target_index:
            rol_sum += len(self.children[cur_index])
            cur_index += 1
        return self.children[cur_index].fetch_at(target_index - rol_sum)

    def pop(
            self, rand: float,
            inv_alpha_norm_priorities: typing.Optional[float] = None
        ) -> typing.Tuple[float, float, Experience]:
        """Pops the experience off of this node, seeded with the given random number between
        0-1. If the random number is chosen uniformly at random on (0, 1), and
        inv_alpha_norm_priorities is not specified, then this selects with probability

        P(i) = (p_i^alpha) / (alpha-norm p).

        Note that this returns (priority, probability, exp), which is the priority of the
        experience returned, the probability that this experience is the one returned (assuming
        rand is unif(0, 1)), and the actual experience itself.
        """
        if not inv_alpha_norm_priorities:
            if self._len == 0:
                raise IndexError('pop from empty node')
            if self._len == 1:
                prob, prio, exp = (1, self.priority, self.child)
                self.priority = 0
                self.alpha_norm_priority = 0
                self._len = 0
                self.child = None
                return prio, prob, exp

            inv_alpha_norm_priorities = 1 / self.alpha_norm_priority

        # maybe it's better to multiply rand by alpha_norm_priority instead of doing it this way?
        # probably only a very minor improvement if it is one, and makes nesting more complicated

        roll_sum = 0
        child_ind = None
        new_amt = None
        child_nonleaf = None
        for ind, child in enumerate(self.children):
            new_amt = child.alpha_norm_priority * inv_alpha_norm_priorities
            if roll_sum + new_amt >= rand:
                if len(child) == 1:
                    child_ind = ind
                    break

                child_nonleaf = child
                break
            roll_sum += new_amt

        if child_nonleaf:
            prio, prob, exp = child_nonleaf.pop(rand - roll_sum, inv_alpha_norm_priorities)
            self._len -= 1
            self.priority -= prio
            self.alpha_norm_priority -= prio ** self.alpha
            return prio, prob, exp

        child = self.children.pop(child_ind)
        self._len -= 1
        self.priority -= child.priority
        self.alpha_norm_priority -= child.alpha_norm_priority

        if self._len == 1:
            self.child = self.children[0].child
            self.children = None
            self._flat = True
        return child.priority, new_amt, child.child


    def __len__(self):
        return self._len

class MemoryPrioritizedReplayBuffer(ReadableReplayBuffer):
    """Loads the entire replay buffer into memory. The sample and next function somewhat
    inefficiently provide a uniform sampling of the replay buffer for compatibility.
    More helpfully, this provides pop which is prioritized as well as add for updating
    the last_td_error.

    This is meant to be analogous to the implementation described in
    https://arxiv.org/pdf/1511.05952.pdf

    Priority calculations:
        priority = |temporal difference| + epsilon
        probability = (priority of item)^(alpha) / (alpha-norm of all priorities)

    Remark:
        To copy the result of this to a new buffer, it's important that you don't just
        loop over the samples. The samples for this, unlike the file readable version,
        do not guarrantee you see each item exactly once per epoch. You can loop over
        this in order by fetch_at or via flatten. Flattening will usually be faster.

    Attributes:
        dirname (str): the path to the directory where we loaded experiences from
        tree (PrioritizedExperienceNode): the experiences in this buffer
        epsilon (float): added to the absolute temporal difference error to ensure
            all experiences have some non-zero probability of being selected
        alpha (float) (from 0 to +inf): 0 means completely uniform selection,
            while +inf means deterministic. Typical values are between 0.4 and 1
        unseen_priority (float): the priority given to things we have not seen before.
            this can be thought of as the td-error that we pretend things have if we
            haven't sent them through the network yet

        history (deque[Experience]): if mark() was called then we begin storing a history
            of returned values from sample() so as to be able to replicate them. We store
            them here
        history_ind (int): where we are currently in the history. If this is the length
            of the history, then new samples are uniform and appended to the history.
            If there are no marks, then this should be 0 (meaning pop from the left).
        marks (list[int]): the list of currently held marks. These are indices in the history.
    """
    def __init__(self, dirname: str, epsilon=1e-6, alpha=0.6, unseen_priority=1.0):
        self.dirname = dirname
        self.tree = PrioritizedExperienceNode(alpha)
        self.epsilon = epsilon
        self.alpha = alpha
        self.unseen_priority = unseen_priority

        self.history = deque()
        self.history_ind = 0
        self.marks = []

        self._load()

    def _load(self):
        """Loads experiences from the file and stores them into the tree. Meant to be invoked only
        once, by the constructor"""
        rb = FileReadableReplayBuffer(self.dirname)
        try:
            for _ in range(len(rb)):
                self.add(next(rb))
        finally:
            rb.close()

    def __next__(self) -> Experience:
        """Selects an experience uniformly at random from this buffer"""
        if self.history:
            if not self.marks:
                return self.history.popleft()
            if self.history_ind == len(self.history):
                tmp = self.tree.fetch_uniform()
                self.history.append(tmp)
                self.history_ind += 1
                return tmp
            tmp = self.history[self.history_ind]
            self.history_ind += 1
            return tmp
        tmp = self.tree.fetch_uniform()
        if self.marks:
            self.history_ind += 1
            self.history.append(tmp)
        return tmp

    def sample(self, num: int) -> typing.List[Experience]:
        """Samples the specified number of experiences uniformly at random from this buffer"""
        return [next(self) for _ in range(num)]

    def fetch_at(self, ind: int) -> Experience:
        """Fetches the experience at index 0 <= ind < len(self). This may be arbitrarily shuffled
        by calls to pop or add.
        """
        return self.tree.fetch_at(ind)

    def flatten(self) -> typing.List[Experience]:
        """Returns a flattened version of this replay buffer in the form of a list."""
        return self.tree.flatten()

    def pop(self) -> typing.Tuple[float, float, Experience]:
        """Returns an experience form this buffer selected according to the relative
        td-error. See class documentation for exact probabilities. You can re-add
        the experience to this buffer with a new last_td_error.

        Returns:
            priority (float): the priority for the selected experience
            probability (float): the probability of the selected Experience being selected
            experience (Experience): the experience that was selected
        """
        return self.tree.pop(random.random())

    def add(self, exp: Experience) -> None:
        """Adds the specified experience to this buffer."""
        if not exp.last_td_error:
            prio = self.unseen_priority
        else:
            prio = abs(exp.last_td_error) + self.epsilon
        self.tree.add(exp, prio)

    def mark(self) -> None:
        """Marks this buffer, which causes all sample and nexts that follow until the next reset
        to be replayed after the reset. This operation may be nested.
        """
        self.marks.append(self.history_ind)

    def reset(self) -> None:
        """Causes this to replay all experiences from __next__ and sample in the same order since
        the last mark
        """
        self.history_ind = self.marks.pop()

    def close(self) -> None:
        """Releases references to the garbage collector. Future operations are undefined"""
        if self.marks:
            import warnings
            warnings.warn('Marks are being leaked', UserWarning)
        self.dirname = None
        self.tree = None
        self.epsilon = None
        self.alpha = None
        self.history = None
        self.history_ind = None
        self.marks = None

    def __len__(self) -> int:
        return len(self.tree)

def merge_buffers(inpaths: typing.List[str], outpath: str) -> None:
    """Merges the replay buffers stored in the inpaths to the
    outpath. The outpath must not already exist"""

    outlen = 0
    outflen = 0
    outlongest_slen = 0
    os.makedirs(outpath)
    with open(os.path.join(outpath, EXPERIENCES_FILE), 'wb') as outfile:
        for inpath in inpaths:
            with open(os.path.join(inpath, META_FILE), 'r') as infile:
                inmeta = json.load(infile)

            inexppath = os.path.join(inpath, EXPERIENCES_FILE)
            if os.path.getsize(inexppath) < inmeta['file_length']:
                raise ValueError(f'{inexppath} is too short (meta file says should be '
                                 + inmeta['file_length'] + ')')

            block = bytearray(inmeta['largest_state_nbytes'])
            blockmv = memoryview(block)
            with open(inexppath, 'rb') as infile:
                while infile.tell() < inmeta['file_length']:
                    nextlen = int.from_bytes(infile.read(4), 'big', signed=False)
                    nread = infile.readinto(blockmv[:nextlen])
                    ctr = 0
                    while nread < nextlen:
                        nread += infile.readinto(blockmv[nread:nextlen])
                        ctr += 1
                        if ctr > 100:
                            raise ValueError('infinite loop detected')

                    outfile.write(nextlen.to_bytes(4, 'big', signed=False))
                    ctr = 0
                    nwritten = outfile.write(blockmv[:nextlen])
                    while nwritten < nextlen:
                        nwritten += outfile.write(blockmv[nwritten:nextlen])
                        ctr += 1
                        if ctr > 100:
                            raise ValueError('infinite loop detected')

                    outlen += 1
                    outflen += 4 + nextlen
                    outlongest_slen = max(outlongest_slen, nextlen)

    with open(os.path.join(outpath, META_FILE), 'w') as outfile:
        json.dump({'length': outlen, 'file_length': outflen,
                   'largest_state_nbytes': outlongest_slen}, outfile)

class ExperienceType:
    """Describes a type of experience. Just a specific type of callable"""
    def __call__(self, exp: Experience) -> bool:
        """Returns True if the given experience is an example of this experience type,
        false otherwise"""
        raise NotImplementedError

class PositiveExperience(ExperienceType):
    """Describes a positive experience"""
    def __call__(self, exp: Experience) -> bool:
        return exp.reward_rec > 0

class NegativeExperience(ExperienceType):
    """Describes a negative experience"""
    def __call__(self, exp: Experience) -> bool:
        return exp.reward_rec <= 0

class ActionExperience(ExperienceType):
    """Describes an experience that involves a particular action"""
    def __init__(self, action: Move):
        self.action = action

    def __call__(self, exp: Experience) -> bool:
        return exp.action == self.action

def balance_experiences(replay_path: str, exp_types: typing.List[ExperienceType],
                        style: str = 'exact') -> int:
    """Loads the replay folder and filters it such that it is balanced according to the
    style.

    Note that balancing experiences may not be possible in all situations - for example, an
    optimal algorithm for a deterministic game will *always* select experiences with positive
    reward, so it is not possible to balance positive and negative rewards.

    However, for a network which is trying to get positive experiences and does achieve some
    positive experiences, it will never not receive any positive experiences. Thus, we can
    always ensure we have at least as many positive experiences as negative experiences.

    Args:
        replay_path (str): the path to the replay folder to balance
        exp_types (typing.List[ExperienceType]): the experience types you want in descending order
        style (str): determines balancing style
            exact: ensures that there are exactly the same number of experience types in each
            desc: ensures that it has at least as many of the first experience type as the second
                experience type, and at least as much of the second experience type as the third,
                etc.

    Returns:
        The number of each type of experience in the dataset
    """

    outpath = os.path.join(replay_path, 'tmp')
    inwriter = FileReadableReplayBuffer(replay_path)
    outwriter = FileWritableReplayBuffer(outpath, exist_ok=True)
    result = 0
    try:
        counters = list(0 for i in range(len(exp_types)))
        for _ in range(len(inwriter)):
            exp: Experience = next(inwriter)
            for i, exp_type in enumerate(exp_types):
                if exp_type(exp):
                    counters[i] += 1
                    break

        if style == 'desc':
            new_counters = [counters[0]]
            for i in range(1, len(exp_types)):
                new_counters.append(min(new_counters[i - 1], counters[i]))
        else:
            new_amt = min(counters)
            new_counters = list(new_amt for _ in exp_types)

        for _ in range(len(inwriter)):
            exp: Experience = next(inwriter)
            typ: int = -1
            for i, exp_type in enumerate(exp_types):
                if exp_type(exp):
                    typ = i
                    break
            if typ >= 0 and new_counters[typ] > 0:
                new_counters[typ] -= 1
                outwriter.add(exp)
    finally:
        outwriter.close()
        inwriter.close()

    os.remove(os.path.join(replay_path, EXPERIENCES_FILE))
    os.remove(os.path.join(replay_path, META_FILE))
    os.rename(os.path.join(outpath, EXPERIENCES_FILE), os.path.join(replay_path, EXPERIENCES_FILE))
    os.rename(os.path.join(outpath, META_FILE), os.path.join(replay_path, META_FILE))
    os.rmdir(outpath)
    return result

def ensure_max_length(replay_path: str, max_exps: int) -> None:
    """Ensures that there are no more than max_exps experiences in the replay buffer at the
    specified location. Experiences are pruned randomly if necessary.
    """

    inwriter = FileReadableReplayBuffer(replay_path)
    if len(inwriter) < max_exps:
        inwriter.close()
        return

    tmppath = os.path.join(replay_path, 'tmp')
    outwriter = FileWritableReplayBuffer(tmppath)

    try:
        for _ in range(max_exps):
            outwriter.add(next(inwriter))
    finally:
        inwriter.close()
        outwriter.close()

    os.remove(os.path.join(replay_path, EXPERIENCES_FILE))
    os.remove(os.path.join(replay_path, META_FILE))
    os.rename(os.path.join(tmppath, EXPERIENCES_FILE), os.path.join(replay_path, EXPERIENCES_FILE))
    os.rename(os.path.join(tmppath, META_FILE), os.path.join(replay_path, META_FILE))
    os.rmdir(tmppath)
