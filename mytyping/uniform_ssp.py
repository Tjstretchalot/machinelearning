"""Describes a uniform word typer that can be used to train without worrying about delays."""

from shared.seqseqprod import SeqSeqProducer, Sequence
import typing
import shared.typeutils as tus
import mytyping.encoding as menc

class UniformSSP(SeqSeqProducer):
    """Describes a simple uniform ssp which just varies the spacing between words by
    a constant amount. Must be given a wordlist.

    Attributes:
        words (list[str]): the words which we can choose from
        char_delay (float): the delay between characters
    """

    def __init__(self, words: typing.List[str], char_delay: int):
        tus.check_list(str, words=words)
        tus.check(char_delay=(char_delay, (int, float)))
        super().__init__(len(words), 1, 2)

        self.words = words
        self.char_delay = float(char_delay)

    def get_current(self) -> typing.Tuple[Sequence, Sequence]:
        word = self.words[self.position]
        inputs = [menc.encode_input(i) for i in word]
        inputs.append(menc.encode_input_stop())
        outputs = [menc.encode_output(i, self.char_delay) for i in word]
        outputs.append(menc.encode_output_stop())

        return Sequence(raw=inputs), Sequence(raw=outputs)

    def get_current_word(self) -> str:
        """Returns the current word"""
        return self.words[self.position]

    def _position(self, pos: int) -> None:
        pass
