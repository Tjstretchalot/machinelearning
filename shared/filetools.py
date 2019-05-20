"""This module produces some helpful tools for deciding where to save things
"""

import sys
import os
import shutil
import typing

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

def recur_unzip(path: str, result: typing.List[str] = None) -> typing.List[str]:
    """Tries to ensure that the specified path exists by extracting zip files that
    exist where directories are missing with the same name. Returns a list of directories
    unzipped in the order that they can be rezipped in

    Args:
        path (str): the path to recursively unzip to
    """

    if result is None:
        result = []

    if os.path.exists(path):
        return result


    my_split = os.path.abspath(path).split(os.path.sep)
    curdir = None
    for thisdir in my_split:
        if curdir is None:
            curdir = thisdir
        else:
            curdir += os.path.sep + thisdir
        if not os.path.exists(curdir):
            if not os.path.exists(curdir + '.zip'):
                zipmany(*result)
                raise FileNotFoundError(curdir)
            result.append(curdir)
            unzip(curdir + '.zip')
    return result


def zipmany(*paths):
    """Zips the specified list of directories in the order that they are given"""
    for path in paths:
        zipdir(path)