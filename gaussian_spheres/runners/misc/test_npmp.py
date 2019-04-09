"""A simple npmp test case"""
import numpy as np
import time
from shared.npmp import NPDigestor
import shared.filetools
import os

def _target(data: np.ndarray, note: str):
    print(f'_target called note={note}, data={data}')
    time.sleep(5)
    print(f'_target(note={note}) finished')

def main():
    """Main runner"""
    dig = NPDigestor('test_npmp', 'gaussian_spheres.runners.misc.test_npmp', '_target', 3)

    for i in range(10):
        data = np.random.uniform(-1, 1, 10)
        note = f'note {i}'
        print(f'calling dig with note {note} and data {data}')
        dig(data=data, note=note)
        print(f'dig call for note {note} finished')

    dig.join()
    print('finished, archiving')
    savepath = shared.filetools.savepath()
    if not os.path.exists(savepath):
        os.makedirs(shared.filetools.savepath())
    if os.path.exists(os.path.join(savepath, 'inps.zip')):
        os.remove(os.path.join(savepath, 'inps.zip'))
    dig.archive_raw_inputs(os.path.join(savepath, 'inps.zip'))
    print('done')

if __name__ == '__main__':
    main()