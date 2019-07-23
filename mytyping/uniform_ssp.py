"""Describes a uniform word typer that can be used to train without worrying about delays."""

from shared.seqseqprod import SeqSeqProducer, Sequence
from shared.perf_stats import PerfStats, NoopPerfStats
import typing
import pytypeutils as tus
import mytyping.encoding as menc

PRE_ENCODE = True
"""If true we cache the encoded versions of words, if false we regenerate.
True = more memory, less cpu. False = less memory, more cpu"""

class UniformSSP(SeqSeqProducer):
    """Describes a simple uniform ssp which just varies the spacing between words by
    a constant amount. Must be given a wordlist.

    Attributes:
        words (list[str]): the words which we can choose from
        char_delay (float): the delay between characters
        encoded_words (list[tuple[Sequence, Sequence]]): the encoded words (if PRE_ENCODE is True)
    """

    def __init__(self, words: typing.List[str], char_delay: int):
        tus.check_listlike(words=(words, str))
        tus.check(char_delay=(char_delay, (int, float)))
        super().__init__(len(words), 1, 2)

        self.words = words
        self.char_delay = float(char_delay)
        self.encoded_words = None

        if PRE_ENCODE:
            self.encoded_words = []
            for word in self.words:
                inputs = [menc.encode_input(i) for i in word]
                inputs.append(menc.encode_input_stop())
                outputs = [menc.encode_output(i, self.char_delay) for i in word]
                outputs.append(menc.encode_output_stop())
                self.encoded_words.append((Sequence(raw=inputs), Sequence(raw=outputs)))


    def get_current(self,
                    perf_stats: PerfStats = NoopPerfStats()) -> typing.Tuple[Sequence, Sequence]:
        if PRE_ENCODE:
            return self.encoded_words[self.position]

        word = self.words[self.position]
        perf_stats.enter('ENCODE_INPUT')
        inputs = [menc.encode_input(i) for i in word]
        inputs.append(menc.encode_input_stop())
        perf_stats.exit_then_enter('ENCODE_OUTPUT')
        outputs = [menc.encode_output(i, self.char_delay) for i in word]
        outputs.append(menc.encode_output_stop())
        perf_stats.exit()

        return Sequence(raw=inputs), Sequence(raw=outputs)

    def get_current_word(self) -> str:
        """Returns the current word"""
        return self.words[self.position]

    def _position(self, pos: int) -> None:
        pass
