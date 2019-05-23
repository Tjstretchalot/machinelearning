"""The goal of this module is to train on USSP through transfer learning. In particular, rather
than training on the entire dataset at once we train a model on a much smaller dataset and then
embed it into a larger network and train on a slightly larger dataset and then repeat this process
"""

import shared.setup_torch # pylint: disable=unused-import
import torch
import mytyping.uniform_ssp as ussp
import shared.ssptrainer as stnr
import shared.trainer as tnr
import mytyping.training as mtnr
import shared.models.seqseq1 as ss1
import mytyping.encoding as menc
import mytyping.wordlist as mwords
import shared.filetools
import os
import logging

SAVEDIR = shared.filetools.savepath()

def _eval(ssp, teacher, network):
    ssp.position = 0
    for _ in range(ssp.remaining_in_epoch):
        true_str = ssp.get_current_word()
        inp, out = next(ssp)

        res = teacher.classify_many(network, [inp])[0]

        if menc.accuracy(res, out) >= 1:
            continue

        print(f'{true_str}<STP> --> ', end='')
        for item in res.raw:
            is_char, key, _ = menc.read_output(item)
            if not is_char:
                print('<STP>')
            else:
                print(key, end='')

def main():
    """Meant to be invoked for this runner"""
    folderpath = os.path.join('out', 'mytyping', 'runners', 'transfer_train_ussp', '1layer', '0', '1')
    words = mwords.load_custom(os.path.join(folderpath, 'words.txt'))
    ssp = ussp.UniformSSP(words.words, 64)

    network = torch.load(os.path.join(folderpath, 'trained_models', 'epoch_800.pt'))

    teacher = ss1.EncoderDecoderTeacher(menc.stop_failer, 30)
    _eval(ssp, teacher, network)
    tracker = mtnr.AccuracyTracker(1, len(words.words), False)

    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    ctx = stnr.SSPGenericTrainingContext(model=network, teacher=teacher, train_ssp=ssp, test_ssp=ssp, optimizers=[], batch_size=1, shared={}, logger=_logger)
    ctx.shared['epochs'] = tnr.EpochsTracker()
    ctx.shared['epochs'].new_epoch = True

    tracker.setup(ctx)
    tracker.pre_loop(ctx)

if __name__ == '__main__':
    main()
