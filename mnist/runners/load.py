"""Tests loading mnist data"""

from mnist.pwl import MNISTData
import numpy as np

def main():
    """Real entry point"""
    data = MNISTData.load_from('data/mnist/train-images-idx3-ubyte', 'data/mnist/train-labels-idx1-ubyte')
    pwl = data.to_pwl()

    print('counts by label:')
    for lbl in range(10):
        cnt = (pwl.real_labels == lbl).sum()
        print(f'  {lbl} = {cnt}')

    print(f'mean pts: {pwl.real_points.mean()}, max={pwl.real_points.max()}')

    quartile = np.percentile(np.abs(pwl.real_points.numpy()), 75)
    print(f' quartile={quartile}')

    pwl = pwl.restrict_to(frozenset(range(10)))

    print('after restricting')

    print('counts by label:')
    for lbl in range(10):
        cnt = (pwl.real_labels == lbl).sum()
        print(f'  {lbl} = {cnt}')

    print(f'mean pts: {pwl.real_points.mean()}, max={pwl.real_points.max()}')
    quartile = np.percentile(np.abs(pwl.real_points.numpy()), 75)
    print(f' quartile={quartile}')

    pwl = pwl.rescale()

    print('after rescaling')

    print(f'mean pts: {pwl.real_points.mean()}, max={pwl.real_points.max()}')
    quartile = np.percentile(np.abs(pwl.real_points.numpy()), 75)
    print(f' quartile={quartile}')

if __name__ == '__main__':
    main()