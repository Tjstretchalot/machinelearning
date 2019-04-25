import os
import time
import shutil
from pathlib import Path

ARCHIVE_PATH = os.path.join(Path.home(), 'archived')

def main():
    nowstr = str(time.time())
    apath = os.path.join(ARCHIVE_PATH, nowstr)
    
    cwd = os.getcwd()
    if os.path.exists('out'):
        print(f'archiving out/ to {apath}')
        shutil.make_archive(apath, 'zip', 'out')
        os.chdir(cwd)
        print(f'cleaning out/')
        shutil.rmtree('out')
        os.chdir(cwd)
    if os.path.exists('tmp'):
        print(f'cleaning tmp/')
        shutil.rmtree('tmp')
        os.chdir(cwd)
    print('all clean')

if __name__ == '__main__':
    main()
