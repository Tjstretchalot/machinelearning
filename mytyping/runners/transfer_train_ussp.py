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

MODEL_NAME = '1layer'
SAVEDIR = os.path.join(shared.filetools.savepath(), MODEL_NAME)

def _eval(ssp, teacher, network):
    ssp.position = 0
    for _ in range(min(ssp.remaining_in_epoch, 10)):
        inp, out = next(ssp)

        print('Presenting ', end='')
        for item in inp.raw:
            is_char, key = menc.read_input(item)
            if not is_char:
                print('<STP>')
            else:
                print(key, end='')


        res = teacher.classify_many(network, [inp])[0]

        print('Got ', end='')
        delays = []
        for item in res.raw:
            is_char, key, delay = menc.read_output(item)
            delays.append(str(delay))
            if not is_char:
                print('<STP>')
            else:
                print(key, end='')

        print('Delays: ' + ', '.join(delays))

def train_on(network, teacher, wordlist, num_words, thisdir, patience):
    """Trains a network with the given settings"""
    thiswords = wordlist.first(num_words)
    thiswords.save(os.path.join(thisdir, 'words.txt'), True)

    ssp = ussp.UniformSSP(words=thiswords.words, char_delay=64)
    trainer = stnr.SSPGenericTrainer(
        train_ssp=ssp,
        test_ssp=ssp,
        teacher=teacher,
        batch_size=1,
        learning_rate=0.003,
        optimizers=[torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=1)],
        criterion=torch.nn.SmoothL1Loss()
    )

    trained_model_dir = os.path.join(thisdir, 'trained_models')
    if os.path.exists(trained_model_dir):
        shared.filetools.deldir(trained_model_dir)

    (trainer
        .reg(tnr.EpochsTracker(verbose=False))
        .reg(tnr.EpochProgress(5, accuracy=True))
        .reg(tnr.DecayTracker())
        .reg(tnr.DecayOnPlateau(patience=patience, verbose=False, initial_patience=5))
        .reg(tnr.DecayStopper(5))
        .reg(tnr.LRMultiplicativeDecayer(reset_state=True))
        .reg(tnr.OnEpochCaller.create_every(tnr.save_model(trained_model_dir), skip=50, suppress_on_inf_or_nan=False))
        .reg(mtnr.AccuracyTracker(5, 100, True, verbose=False))
    )

    result = trainer.train(network)

    _eval(ssp, teacher, network)

    return result['accuracy']

def main():
    """Meant to be invoked for this runner"""
    start_block = 0
    num_blocks = 50
    block_size = 10
    acc_low_changes = (
        (1, 0, 1),
        (1, 0, 1),
        (1, 0, 1),
        (0, 1, 0)
    )
    acc_low_ch_ind = 0
    patience = 45

    total_wordlist = mwords.load_custom('data/commonwords/google-10000-english-no-swears.txt').subset(num_blocks * block_size)
    if len(total_wordlist.words) != num_blocks * block_size:
        raise ValueError(f'not enough words (got only {len(total_wordlist.words)}, not {num_blocks * block_size})')

    if start_block == 0:
        network = ss1.EncoderDecoder(
            input_dim=menc.INPUT_DIM,
            encoding_dim=8,
            context_dim=4,
            decoding_dim=8,
            output_dim=menc.OUTPUT_DIM,
            encoding_layers=1,
            decoding_layers=1
        )
    else:
        network = torch.load(os.path.join(SAVEDIR, str(start_block), 'trained_models', 'epoch_finished.pt'))

    teacher = ss1.EncoderDecoderTeacher(menc.stop_failer, 30)
    for block in range(start_block, num_blocks):
        folder = os.path.join(SAVEDIR, str(block))
        num_words = block_size * (block + 1)
        acc = train_on(network, teacher, total_wordlist, num_words, folder, patience)
        while acc < 0.95:
            enc_amt, ctx_amt, dec_amt = acc_low_changes[acc_low_ch_ind]
            acc_low_ch_ind = (acc_low_ch_ind + 1) % len(acc_low_changes)
            print(f'Network size increasing by {enc_amt}, {ctx_amt}, {dec_amt}')
            network = network.transfer_up(network.encoding_dim + enc_amt,
                                          network.context_dim + ctx_amt,
                                          network.decoding_dim + dec_amt)
            acc = train_on(network, teacher, total_wordlist, num_words, folder, patience)

if __name__ == '__main__':
    main()
