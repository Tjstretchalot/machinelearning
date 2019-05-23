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

    trainer.train(network)

    _eval(ssp, teacher, network)

def main():
    """Meant to be invoked for this runner"""
    num_blocks = 50
    block_size = 10
    small_block_size = 5
    small_num_blocks = block_size / small_block_size
    start_block = 0 # set to 0 for a clean model
    start_small_block = 0 # set to 0 for a clean model
    encoding_dim_exp = [1, 2, 4, 1, 0] * ((num_blocks // block_size) + 1)
    context_dim_exp = [1, 0, 0, 0, 3] * ((num_blocks // block_size) + 1)
    decoding_dim_exp = [1, 2, 4, 1, 0] * ((num_blocks // block_size) + 1)
    patience = [45 for _ in range(num_blocks)]

    if small_num_blocks != int(small_num_blocks):
        raise ValueError(f'small_num_blocks={small_num_blocks}')
    small_num_blocks = int(small_num_blocks)

    total_wordlist = mwords.load_custom('data/commonwords/google-10000-english-no-swears.txt').subset(num_blocks * block_size)
    if len(total_wordlist.words) != num_blocks * block_size:
        raise ValueError(f'not enough words (got only {len(total_wordlist.words)}, not {num_blocks * block_size})')

    if start_block == 0 and start_small_block == 0:
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
        network = torch.load(os.path.join(SAVEDIR, str(start_block), str(start_small_block), 'trained_models', 'epoch_finished.pt'))

    teacher = ss1.EncoderDecoderTeacher(menc.stop_failer, 30)
    for large_block in range(start_block, num_blocks+1):
        if large_block == start_block:
            miter = range(start_small_block + 1, small_num_blocks)
        else:
            miter = range(small_num_blocks)
        for small_block in miter:
            folder = os.path.join(SAVEDIR, str(large_block), str(small_block))
            num_words = large_block * block_size + small_block * small_block_size
            train_on(network, teacher, total_wordlist, num_words, folder, patience[large_block])

        enc_amt = encoding_dim_exp[large_block]
        ctx_amt = context_dim_exp[large_block]
        dec_amt = decoding_dim_exp[large_block]
        print(f'Network size increasing by {enc_amt}, {ctx_amt}, {dec_amt}')
        network = network.transfer_up(network.encoding_dim + enc_amt,
                                      network.context_dim + ctx_amt,
                                      network.decoding_dim + dec_amt)

if __name__ == '__main__':
    main()
