"""Trains a single network on the uniform ssp style"""

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

SAVEDIR = shared.filetools.savepath()

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

def main():
    """Meant to be invoked for this runner"""
    words = mwords.load_custom('data/commonwords/google-10000-english-no-swears.txt').subset(50)
    ssp = ussp.UniformSSP(words=words.words, char_delay=64)

    network = ss1.EncoderDecoder(
        input_dim=menc.INPUT_DIM,
        encoding_dim=64,
        context_dim=32,
        decoding_dim=64,
        output_dim=menc.OUTPUT_DIM,
        encoding_layers=1,
        decoding_layers=1
    )

    teacher = ss1.EncoderDecoderTeacher(menc.stop_failer, 30)

    trainer = stnr.SSPGenericTrainer(
        train_ssp=ssp,
        test_ssp=ssp,
        teacher=teacher,
        batch_size=1,
        learning_rate=0.003,
        optimizers=[torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=1)],
        criterion=torch.nn.MSELoss()
    )

    trained_model_dir = os.path.join(SAVEDIR, 'trained_models')
    if os.path.exists(trained_model_dir):
        shared.filetools.deldir(trained_model_dir)

    (trainer
     .reg(tnr.EpochsTracker(verbose=False))
     .reg(tnr.EpochProgress(5))
     .reg(tnr.DecayTracker())
     .reg(tnr.DecayOnPlateau(patience=15, verbose=False))
     .reg(tnr.DecayStopper(5))
     .reg(tnr.LRMultiplicativeDecayer(reset_state=True))
     .reg(tnr.OnEpochCaller.create_every(tnr.save_model(trained_model_dir), skip=50, suppress_on_inf_or_nan=False))
     .reg(mtnr.AccuracyTracker(5, 100, True))
    )

    trainer.train(network)
    print('finished')

    _eval(ssp, teacher, network)

if __name__ == '__main__':
    main()
