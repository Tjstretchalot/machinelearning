"""This module produces some helpful tools for deciding where to save things
"""

import sys
import os
import shutil

def savepath():
    """Gets the best savepath for the current script. The result
    is a directory.
    """
    script_path = sys.argv[0]
    base_path = os.getcwd()

    if not script_path.startswith(base_path):
        raise OSError(f'This trick only works consistently when called with -m convention from the root directory (sys.argv[0] = {sys.argv[0]}, os.getcwd() = {os.getcwd()}')

    module_path = script_path[len(base_path)+1:]
    module_path = os.path.splitext(module_path)[0]
    return os.path.join('out', module_path)

def deldir(dirpath: str):
    """Deletes the specified directory

    Args:
        dirpath (str): the path to the directory that is deleted
    """

    if not isinstance(dirpath, str):
        raise ValueError(f'expected dirpath is str, got {dirpath} (type={type(dirpath)})')
    if not os.path.exists(dirpath):
        return

    cwd = os.getcwd()
    shutil.rmtree(dirpath)
    os.chdir(cwd)


def zipdir(dirpath: str):
    """Zips the specified directory and deletes it.

    Args:
        dirpath (str): the path to the directory that is zipped then deleted
    """

    if not isinstance(dirpath, str):
        raise ValueError(f'expected dirpath is str, got {dirpath}')
    if not os.path.exists(dirpath):
        raise FileNotFoundError(f'cannot zip {dirpath} (doesnt exist)')
    if os.path.exists(dirpath + '.zip'):
        raise FileExistsError(f'cannot zip {dirpath} (zip already exists)')

    cwd = os.getcwd()
    shutil.make_archive(dirpath, 'zip', dirpath)
    os.chdir(cwd)
    shutil.rmtree(dirpath)
    os.chdir(cwd)

def unzip(archivepath: str):
    """Unzips the given archive and deletes it

    Args:
        archivepath (str): the archive to unzip
    """

    if not isinstance(archivepath, str):
        raise ValueError(f'expected archivepath is str, got {archivepath}')
    if not os.path.exists(archivepath):
        raise ValueError(f'expected {archivepath} exists to extract, but does not')

    wo_ext = os.path.splitext(archivepath)[0]
    if wo_ext == archivepath:
        raise ValueError(f'expected {archivepath} has archive extension, but does not')
    if os.path.exists(wo_ext):
        raise ValueError(f'expected {wo_ext} does not exist to extract {archivepath}, but it does')

    cwd = os.getcwd()
    shutil.unpack_archive(archivepath, wo_ext)
    os.chdir(cwd)
    os.remove(archivepath)
