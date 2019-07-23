"""Capable of downloading and loading the TWL06 Scrabble Word List"""

import typing
import pytypeutils as tus
import random
import os
import hashlib
import numpy as np

FILEURL = 'https://www.wordgamedictionary.com/twl06/download/twl06.txt'
FILEHASH = '3c266d1d0131e6ec0aa7268f3f171b84'

class WordList:
    """Describes the TWL06 word list with any helpful functions

    Attributes:
        words (list[str]): the actual list of words
    """

    def __init__(self, words: typing.List[str]):
        tus.check_listlike(words=(words, str))
        self.words = words

    def permute(self) -> 'WordList':
        """Randomly permutes the word list"""
        random.shuffle(self.words)
        return self

    def subset(self, num) -> 'WordList':
        """Selects a random subset of at most the given size"""
        if num > len(self.words):
            return self.permute()

        inds = np.random.choice(len(self.words), num)
        new_words = []
        for i in inds:
            new_words.append(self.words[i])
        return WordList(new_words)

    def first(self, num) -> 'WordList':
        """Selects the first num words in the list"""
        return WordList(self.words[:num].copy())

    def save(self, outpath: str, exist_ok: bool) -> None:
        """Saves this wordlist to the given path"""
        tus.check(outpath=(outpath, str), exist_ok=(exist_ok, bool))

        if not exist_ok and os.path.exists(outpath):
            raise FileExistsError(outpath)
        dirname = os.path.dirname(outpath)
        os.makedirs(dirname, exist_ok=True)
        with open(outpath, 'w') as outfile:
            for word in self.words:
                print(word, file=outfile)



def verify(path: str) -> bool:
    """Verifies that the given path actually contains the wordlist

    Arguments:
        path (str): the path you want to verify

    Returns:
        True if the path exists and contains the wordlist, False if it does not exist or is not the
        wordlist
    """

    if not os.path.exists(path):
        return False
    hash_md5 = hashlib.md5()
    with open(path, 'rb') as infile:
        for chunk in iter(lambda: infile.read(4096), b''):
            hash_md5.update(chunk)
    hexed = hash_md5.hexdigest()

    if hexed != FILEHASH:
        import warnings
        warnings.warn(f'Path {path} exists but is not the TWL06 database; expected hash is \'{FILEHASH}\', got \'{hexed}\'')
        return False
    return True

def load(path: str, download=True) -> WordList:
    """Loads the TWL06 wordlist from the given path, downloading it if it doesn't exist

    Arguments:
        path (str): the path to the file to download
        download (bool): True to automatically download if it doesn't exist
    """

    if not verify(path):
        if os.path.exists(path):
            os.remove(path)
        if not download:
            raise FileNotFoundError(path)
        pathdir = os.path.dirname(path)
        os.makedirs(pathdir, exist_ok=True)

        print(f'Downloading TWL06 wordlist to {path}')

        import requests
        req = requests.get(FILEURL, headers={'User-Agent': 'tjstretchalot-machinelearning-mtimothy984@gmail.com'})
        with open(path, 'wb') as infile:
            infile.write(req.content)

        if not verify(path):
            raise Exception(f'after downloading hash didnt match!')

    words = []
    with open(path, 'r') as infile:
        infile.readline()
        infile.readline()
        for line in infile:
            words.append(line.strip())
    return WordList(words)

def load_custom(path: str) -> WordList:
    """Loads a custom word list with 1 word per line from the given path"""
    with open(path, 'r') as infile:
        words = [line.strip() for line in infile]
    return WordList(words)
