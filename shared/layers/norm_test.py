"""Can be run to verify the norm layers work as expected"""
import shared.setup_torch # pylint: disable=unused-import
import torch
import shared.layers.norm as norm

def main():
    """Main entry for running norm layer tests"""
    _evaluative_test(5)
    _fuzz_test(1)
    _fuzz_test(1000)

def _evaluative_test(features: int):
    means = torch.randn(features)
    stds = torch.randn(features)

    lyr = norm.EvaluatingAbsoluteNormLayer(features, means, 1 / stds)

    inp = torch.randn((32, features))

    truth = (inp - means) / stds
    got = lyr(inp)

    if not torch.allclose(truth, got):
        print(f'means={means}')
        print(f'stds={stds}')
        print(f'inp[0]={inp[0]}')
        print(f'exp inp[0] -> {truth[0]}')
        print(f'got inp[0] -> {got[0]}')
        raise ValueError(f'expected {truth}, got {got}')


def _fuzz_test(features: int, samples: int = 1024, batch: int = 32):
    samples = batch * (samples // batch)
    lyr = norm.LearningAbsoluteNormLayer(features)

    inps = torch.randn((samples, features))
    inps *= torch.randn(features)
    inps += torch.randn(features)

    means = inps.mean(dim=0)
    stds = inps.std(dim=0)

    for i in range(0, samples, batch):
        lyr(inps[i:i+batch])

    as_eval = lyr.to_evaluative()
    if not torch.allclose(means, as_eval.means):
        raise ValueError(f'means are off: expected {means}, got {as_eval.means}')
    if not torch.allclose(stds, 1 / as_eval.inv_std):
        raise ValueError(f'stds are off: expected {stds}, got {1 / as_eval.inv_std} (recip exp: {1 / stds})')

if __name__ == '__main__':
    main()
