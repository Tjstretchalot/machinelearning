"""This is a storage measure - it stores the activations of the network in both
a python and matlab format.
"""

import numpy as np
import torch
import scipy.io
import typing
import os
import json
import time

from shared.trainer import GenericTrainingContext
import shared.filetools as filetools
import shared.measures.utils as mutils
import shared.npmp as npmp

SAVE_SPLIT = False
"""If true, then save_using will save each array individually alongside the all array"""

def save_using(samples: np.ndarray, labels: np.ndarray, *layer_acts: typing.Tuple[np.ndarray],
               num_labels: int, outpath: str, exist_ok: bool, meta: dict,
               **additional: typing.Dict[str, np.ndarray]):
    """Stores the activations of the network to the given file, optionally
    overwriting it if it already exists.

    Args:
        samples (np.ndarray): the samples presented to the network of dimensions
            [num_samples, input_dim]
        labels (np.ndarray): the labels corresponding to the samples presented
            [num_samples]
        layer_acts (tuple[np.ndarray]): the activations of the network. each element
            corresponds to an array of activations with dimensions
            [num_samples, layer_size]
        outpath (str): the file to save to, should be a zip file
        exist_ok (bool): True to overwrite existing files, False not to
        meta (dict): saved alongside the data in json-format
        additional (dict[str, ndarray]): any additional arrays to save
    """
    filepath, folderpath = mutils.process_outfile(outpath, exist_ok)

    os.makedirs(folderpath, exist_ok=True)

    label_masks = [labels == val for val in range(num_labels)]

    asdict = dict({'samples': samples, 'labels': labels}, **additional)
    for layer, act in enumerate(layer_acts):
        asdict[f'layer_{layer}'] = act
        for label, mask in enumerate(label_masks):
            asdict[f'layer_{layer}_label_{label}'] = act[mask]
    scipy.io.savemat(os.path.join(folderpath, 'all'), asdict) # pylint: disable=no-member
    np.savez(os.path.join(folderpath, 'all'), **asdict)

    if SAVE_SPLIT:
        for key, val in asdict.items():
            scipy.io.savemat(os.path.join(folderpath, key), {key: val}) # pylint: disable=no-member
            np.savez(os.path.join(folderpath, key), val)

    scipy.io.savemat(os.path.join(folderpath, 'meta'), meta) # pylint: disable=no-member
    with open(os.path.join(folderpath, 'meta.json'), 'w') as outfile:
        json.dump(meta, outfile)

    if os.path.exists(filepath):
        os.remove(filepath)
    filetools.zipdir(folderpath)

def merge_many(outpath: str, *paths, auto_open_zips=True):
    """Merges the result of multiple runs. Copies select npz and mat files from
    the given folders, stacks them on dimension 0, and then outputs them to the
    specified output path.

    Example:
        merge_many('repeats/all/epoch_finished', *[f'repeats/repeat{i}/epoch_finished' for i in range(10)])

    Args:
        outpath (str): where the final merged files are stored
        paths (tuple[str]): the paths passed to save_using that will be merged
        auto_open_zips (bool, default True): if True then if we come across a directory that
            doesn't exist in paths we will check if the corresponding zip does exist. If so,
            we extract, fetch, and rezip
    """

    if not paths:
        raise ValueError(f'must have at least one path!')

    if os.path.exists(outpath):
        filetools.deldir(outpath)

    os.makedirs(outpath)

    cur_all = None
    for path in paths:
        to_rezip = []
        if auto_open_zips:
            to_rezip = filetools.recur_unzip(path)

        with np.load(os.path.join(path, 'all.npz')) as allnp:
            if cur_all is None:
                cur_all = dict()
                for k in allnp.keys():
                    cur_all[k] = np.expand_dims(allnp[k], 0)
            else:
                cur_all: dict
                for k in allnp.keys():
                    if k in cur_all:
                        if allnp[k].shape != cur_all[k].shape[1:]:
                            print(f'allnp[{k}].shape = {allnp[k].shape}, cur_all[k].shape = {cur_all[k].shape}; path={path}')
                        cur_all[k] = np.concatenate((cur_all[k], np.expand_dims(allnp[k], 0)), axis=0)

                for k in cur_all:
                    if k not in allnp.keys():
                        del cur_all[k]

        if auto_open_zips:
            filetools.zipmany(*to_rezip)

    scipy.io.savemat(os.path.join(outpath, 'all'), cur_all) # pylint: disable=no-member
    np.savez(os.path.join(outpath, 'all'), **cur_all)

    if SAVE_SPLIT:
        for key, val in cur_all.items():
            scipy.io.savemat(os.path.join(outpath, key), {key: val}) # pylint: disable=no-member
            np.savez(os.path.join(outpath, key), val)

def during_training(savepath: str, dig: npmp.NPDigestor, num_points=3000, meta: dict = None):
    """Returns a callable that saves activations to the given file hint. This
    will also save the model.

    Args:
        savepath (str): the folder to save activations to
        dig (npmp.NPDigestor): the digestor to use for the file io
        num_points (int): maximum number of points to run through the network
        meta (dict, optional): if specified it is stored alongside the saved activations
    """
    if os.path.exists(savepath):
        raise ValueError(f'[ACTS] {savepath} already exists')

    if meta is None:
        meta = {'time': time.time()}

    points_by_pwlname = dict()
    labels_by_pwlname = dict()
    def on_step(context: GenericTrainingContext, fname_hint: str):
        context.logger.info('[ACTS] Storing hidden activations (hint: %s)', fname_hint)

        pwls = [('train', context.train_pwl)]
        if context.test_pwl != context.train_pwl:
            pwls.append(('test', context.test_pwl))

        for pwlname, pwl in pwls:
            nump = min(pwl.epoch_size, num_points)

            if pwlname not in points_by_pwlname:
                points = torch.zeros((nump, pwl.input_dim), dtype=context.points.dtype)
                labels = torch.zeros((nump,), dtype=context.labels.dtype)

                pwl.mark()
                if hasattr(pwl, 'fill_uniform'):
                    pwl.fill_uniform(points, labels)
                else:
                    pwl.position = 0
                    pwl.fill(points, labels)
                pwl.reset()
                points_by_pwlname[pwlname] = points
                labels_by_pwlname[pwlname] = labels
            else:
                points = points_by_pwlname[pwlname]
                labels = labels_by_pwlname[pwlname]

            hidacts = mutils.get_hidacts_with_sample(context.model, points, labels)
            hidacts: mutils.NetworkHiddenActivations

            points, labels = None, None
            hidacts.numpy()

            additional = dict()
            additional['epoch'] = np.array([context.shared['epochs'].epoch], dtype='float64')
            if 'accuracy' in context.shared:
                acctracker = context.shared['accuracy']
                if acctracker.last_measure_epoch != context.shared['epochs'].epoch:
                    context.logger.debug('[ACTS] Forcing accuracy measure')
                    acctracker.measure(context)
                additional['accuracy'] = np.array([acctracker.accuracy], dtype='float64')
                context.logger.debug('[ACTS] Detected and included accuracy')


            dig(hidacts.sample_points, hidacts.sample_labels, *hidacts.hid_acts,
                num_labels=pwl.output_dim, outpath=os.path.join(savepath, fname_hint, pwlname),
                exist_ok=False, meta=meta, target_module='shared.measures.acts',
                target_name='save_using')
    return on_step


