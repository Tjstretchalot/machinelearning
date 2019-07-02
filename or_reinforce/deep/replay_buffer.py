"""This is used for the replay buffer of experiences. The replay buffer is well-described
in this article: https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4
"""
import typing
import os
import json
import struct
import io
import random

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
    """
    def __init__(self, state: GameState, action: Move, delay: int, new_state: GameState,
                 reward_rec: float, player_id: int):
        tus.check(state=(state, GameState), action=(action, Move), delay=(delay, int),
                  reward_rec=(reward_rec, float), player_id=(player_id, int))

        self.state = state
        self.action = action
        self.delay = delay
        self.new_state = new_state
        self.reward_rec = reward_rec
        self.player_id = player_id

    def has_custom_serializer(self):
        return True

    def to_prims(self) -> bytes:
        arr = io.BytesIO()
        serd_state = self.state.to_prims()
        arr.write(len(serd_state).to_bytes(4, 'big', signed=False))
        arr.write(serd_state)
        arr.write(int(self.action).to_bytes(4, 'big', signed=False))
        arr.write(self.delay.to_bytes(4, 'big', signed=False))
        serd_state = self.new_state.to_prims()
        arr.write(len(serd_state).to_bytes(4, 'big', signed=False))
        arr.write(serd_state)
        arr.write(struct.pack('>f', self.reward_rec))
        arr.write(self.player_id.to_bytes(4, 'big', signed=False))
        return arr.getvalue()

    @classmethod
    def from_prims(cls, prims: bytes) -> 'Experience': # pylint: disable=line-too-long, arguments-differ
        arr = io.BytesIO(prims)
        arr.seek(0, 0)
        state_len = int.from_bytes(arr.read(4), 'big', signed=False)
        state = GameState.from_prims(arr.read(state_len))
        action = Move(int.from_bytes(arr.read(4), 'big', signed=False))
        delay = int.from_bytes(arr.read(4), 'big', signed=False)
        state_len = int.from_bytes(arr.read(4), 'big', signed=False)
        new_state = GameState.from_prims(arr.read(state_len))
        reward = struct.unpack('>f', arr.read(4))[0]
        player_id = int.from_bytes(arr.read(4), 'big', signed=False)
        return Experience(state, action, delay, new_state, reward, player_id)

    def __eq__(self, other: 'Experience') -> bool:
        if not isinstance(other, Experience):
            return False
        if self.state != other.state:
            return False
        if self.action != other.action:
            return False
        if self.delay != other.delay:
            return False
        if self.new_state != other.new_state:
            return False
        if abs(self.reward_rec - other.reward_rec) > 1e-6:
            return False
        if self.player_id != other.player_id:
            return False
        return True

    def __repr__(self) -> str:
        return (f'Experience [state={repr(self.state)}, action={repr(self.action)}, '
                + f'new_state={repr(self.new_state)}, '
                + f'reward_rec={repr(self.reward_rec)}, player_id={self.player_id}]')

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

def balance_experiences(replay_path: str, exp_types: typing.List[ExperienceType]) -> int:
    """Loads the replay folder and filters it such that it has at least as many of the
    first experience type as the second experience type, and at least as much of the second
    experience type as the third, etc.

    Note that balancing experiences may not be possible in all situations - for example, an
    optimal algorithm for a deterministic game will *always* select experiences with positive
    reward, so it is not possible to balance positive and negative rewards.

    However, for a network which is trying to get positive experiences and does achieve some
    positive experiences, it will never not receive any positive experiences. Thus, we can
    always ensure we have at least as many positive experiences as negative experiences.

    Args:
        replay_path (str): the path to the replay folder to balance
        exp_types (typing.List[ExperienceType]): the experience types you want in descending order

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

        new_counters = [counters[0]]
        for i in range(1, len(exp_types)):
            new_counters.append(min(new_counters[i - 1], counters[i]))

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
