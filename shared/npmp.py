"""A numpy-targetted multiprocessing module which is assumed to store its result
to file. The usage looks very similar to calling the target function on a separate
process while also releasing the memory on the main thread. Also passes json serializable
arguments
"""
import psutil
import warnings
import os
import shutil
import typing
import json
import importlib
import time
import traceback
import sys
from multiprocessing import Process
import numpy as np
from shared.filetools import zipdir

_PRIMS = (bool, int, float, str)

def _get_working_dir(iden: str):
    return os.path.join('tmp', 'npmp', iden)

def _worker_target(identifier: str, worker_id: int, target_module: str, target_name: str):
    mod = importlib.import_module(target_module)
    target = getattr(mod, target_name)

    worker_path = os.path.join(_get_working_dir(identifier), str(worker_id))
    numpy_path = os.path.join(worker_path, 'numpy.npz')
    json_path = os.path.join(worker_path, 'other.json')

    prims_raw: dict
    with open(json_path, 'r') as infile:
        prims_raw = json.load(infile)

    if not 'num_args' in prims_raw:
        raise ValueError(f'expected num_args in prims_raw but it is not')
    num_args = prims_raw['num_args']
    del prims_raw['num_args']
    if not isinstance(num_args, int):
        raise ValueError(f'expected num_args is int, got {num_args} (type={type(num_args)})')

    numpy_raw = dict()
    with np.load(numpy_path) as indata:
        for key, val in indata.items():
            numpy_raw[key] = val

    args = []
    kwargs = dict()

    for argn in range(num_args):
        sargn = f'marg_{argn}'
        if sargn in prims_raw:
            args.append(prims_raw[sargn])
            del prims_raw[sargn]
        else:
            args.append(numpy_raw[sargn])
            del numpy_raw[sargn]

    for k, v in prims_raw.items():
        kwargs[k] = v
    for k, v in numpy_raw.items():
        kwargs[k] = v

    try:
        starttime = time.time()
        target(*args, **kwargs)
        duration = time.time() - starttime
        if duration > 1:
            print(f'[NPDigestor] finished long process {target_module}.{target_name} in {duration:.3f}s')
    except:
        print(f'[NPDigestor] Error while handling target {target_module}.{target_name}: ', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

class NPDigestor:
    """The wrapper class that should be used instead of calling the function directly.

    Attributes:
        identifier (str): a unique identifier to this digestor
        workers_spawned (int): the number of workers spawned in total
        workers (list[Process]): the workers that were started but have not confirmed
            to have finished yet
        max_workers (int): the maximum number of workers out at one time
        target_module (str, optional): the module path (i.e., 'shared.pca') which will be invoked
        target_name (str, optional): the name of the function (i.e. 'plot_pc_trajectory') which
            is in the target module and will be invoked

        _prepared (bool): if we have prepare() d
    """

    def __init__(self, identifier: str, max_workers: int,
                 target_module: typing.Optional[str] = None,
                 target_name: typing.Optional[str] = None):
        if not isinstance(identifier, str):
            raise ValueError(f'expected identifier is str, got {identifier} (type={type(identifier)})')
        if target_module is not None and not isinstance(target_module, str):
            raise ValueError(f'expected target_module is str, got {target_module} (type={type(target_module)})')
        if target_name is not None and not isinstance(target_name, str):
            raise ValueError(f'expected target_name is str, got {target_name} (type={type(target_name)})')
        if not isinstance(max_workers, int):
            raise ValueError(f'expected max_workers is int, got {max_workers} (type={type(max_workers)})')
        if max_workers <= 0:
            raise ValueError(f'expected a positive number of workers, got {max_workers}')

        phys_cpus = psutil.cpu_count(logical=False)
        if max_workers > phys_cpus:
            warnings.warn(f'too many workers suggested ({max_workers}) compared to cores ({phys_cpus}) (auto decreased)', UserWarning)
            max_workers = phys_cpus

        self.identifier = identifier
        self.target_module = target_module
        self.target_name = target_name
        self.max_workers = max_workers

        self.workers = []
        self.workers_spawned = 0

        self._prepared = False

    def prepare(self):
        """Does not need to be called externally. This verifies that working folders are
        available.
        """
        if self._prepared:
            return
        self._prepared = True

        working = _get_working_dir(self.identifier)
        if os.path.exists(working):
            cwd = os.getcwd()
            shutil.rmtree(working)
            os.chdir(cwd)

        os.makedirs(working)

    def _check_container(self, cont):
        stack = []
        seen = set()
        seen_unhashs = [] # some not hashable sadly
        stack.append(cont)
        while stack:
            cur = stack.pop()
            if not isinstance(cur, (dict, list)):
                seen.add(cur)
            else:
                seen_unhashs.append(cur)

            if isinstance(cur, (list, tuple, dict)):
                iterable = cur.values() if isinstance(cur, dict) else cur
                for val in iterable:
                    if isinstance(val, _PRIMS):
                        continue
                    if isinstance(val, (dict, list)):
                        if val in seen_unhashs:
                            raise ValueError(f'found recursive reference {val}')
                    elif val in seen:
                        raise ValueError(f'found recursive reference {val}')
                    stack.append(val)
            else:
                raise ValueError(f'unknown type {cur}')

    def separate(self, *args, **kwargs) -> typing.Tuple[str, str, dict, dict]:
        """Prepares the specified arguments to be passed to a different thread
        by separating them into two dictionaries; one of which has values which
        are numpy arrays and the other which is json serializable. May pass two
        special keywords: target_module and target_name. If these are provided
        then they are used instead of the digestors defaults.
        """
        target_module = self.target_module
        target_name = self.target_name

        if 'target_module' in kwargs:
            target_module = kwargs['target_module']
            if not isinstance(target_module, str):
                raise ValueError(f'expected target_module is str, got {target_module}')
            del kwargs['target_module']
        if 'target_name' in kwargs:
            target_name = kwargs['target_name']
            if not isinstance(target_name, str):
                raise ValueError(f'expected target_name is str, got {target_name}')
            del kwargs['target_name']

        numpy_ready = dict()
        json_ready = dict()

        for argn, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                numpy_ready[f'marg_{argn}'] = arg
            elif isinstance(arg, _PRIMS):
                json_ready[f'marg_{argn}'] = arg
            else:
                try:
                    self._check_container(arg)
                except Exception as exc:
                    raise ValueError(f'failed to separate arg {argn} = {arg}') from exc

                json_ready[f'marg_{argn}'] = arg

        for argn, arg in kwargs.items():
            if not isinstance(argn, str):
                raise ValueError(f'expected kwargs argn={argn} is str, but is {type(argn)}')
            if argn.startswith('marg_') or argn == 'num_args':
                raise ValueError(f'reserved arg name: {argn}')

            if isinstance(arg, np.ndarray):
                numpy_ready[argn] = arg
            elif isinstance(arg, _PRIMS):
                json_ready[argn] = arg
            else:
                try:
                    self._check_container(arg)
                except Exception as exc:
                    raise ValueError(f'failed to separate kwarg {argn}') from exc

                json_ready[argn] = arg

        json_ready['num_args'] = len(args)
        return target_module, target_name, numpy_ready, json_ready

    def _setup_worker_files(self, worker_id: int, numpy_ready: dict, json_ready: dict,
                            target_module: str, target_name: str):
        worker_folder = os.path.join(_get_working_dir(self.identifier), str(worker_id))
        os.makedirs(worker_folder)

        numpy_file = os.path.join(worker_folder, 'numpy.npz')
        json_file = os.path.join(worker_folder, 'other.json')
        meta_file = os.path.join(worker_folder, 'meta.json')

        np.savez_compressed(numpy_file, **numpy_ready)
        with open(json_file, 'w') as outfile:
            json.dump(json_ready, outfile)
        with open(meta_file, 'w') as outfile:
            json.dump({'target_module': target_module, 'target_name': target_name}, outfile)

    def _spawn(self, worker_id: int, target_module: str, target_name: str) -> Process:
        if not isinstance(target_module, str):
            raise ValueError(f'expected target_module is str, got {target_module}')
        if not isinstance(target_name, str):
            raise ValueError(f'expected target_name is str, got {target_name}')

        proc = Process(target=_worker_target,
                       args=(self.identifier, worker_id, target_module, target_name))
        proc.daemon = True
        proc.start()
        return proc

    def _prune(self):
        for i in range(len(self.workers) - 1, -1, -1):
            if not self.workers[i].is_alive():
                del self.workers[i]

    def _invoke_blocking(self, worker_id: int, target_module: str, target_name: str):
        self._prune()
        while len(self.workers) >= self.max_workers:
            time.sleep(0.01)
            self._prune()

        self.workers.append(self._spawn(worker_id, target_module, target_name))

    def __call__(self, *args, **kwargs):
        if not self._prepared:
            self.prepare()

        worker_id = self.workers_spawned
        self.workers_spawned += 1

        target_module, target_name, numpy_ready, json_ready = self.separate(*args, **kwargs)
        self._setup_worker_files(worker_id, numpy_ready, json_ready, target_module, target_name)

        self._invoke_blocking(worker_id, target_module, target_name)

    def repeat_raw(self, inpath: str, target_module: typing.Optional[str] = None,
                   target_name: typing.Optional[str] = None):
        """Takes a path to a folder that was archived and reruns it.

        Args:
            inpath (str): the path to the input folder
            target_module (str, optional): Defaults to None. If specified, the callable to invoke,
                otherwise if the meta.json file exists that is used, otherwise
                self.target_module is used
            target_name (str, optional): Defaults to None. If specified, the callable to invoke,
                otherwise if the meta.json file exists that is used, otherwise
                self.target_name is used
        """
        if target_module is None or target_name is None:
            metapath = os.path.join(inpath, 'meta.json')
            if os.path.exists(metapath):
                with open(metapath, 'r') as infile:
                    meta = json.load(infile)
                if target_module is None:
                    target_module = meta['target_module']
                if target_name is None:
                    target_name = meta['target_name']
            else:
                if target_module is None:
                    target_module = self.target_module
                if target_name is None:
                    target_name = self.target_name

        if not isinstance(inpath, str):
            raise ValueError(f'expected inpath is str, got {inpath}')
        if not isinstance(target_module, str):
            raise ValueError(f'expected target_module is str, got {target_module}')
        if not isinstance(target_name, str):
            raise ValueError(f'expected target_name is str, got {target_name}')

        if not os.path.exists(inpath):
            raise FileNotFoundError(f'cannot repeat {inpath} (file does not exist)')
        if not os.path.isdir(inpath):
            raise ValueError(f'cannot repeat {inpath} (dir expected but is not dir)')

        worker_id = self.workers_spawned
        self.workers_spawned += 1

        worker_folder = os.path.join(_get_working_dir(self.identifier), str(worker_id))
        cwd = os.getcwd()
        shutil.copytree(inpath, worker_folder)
        os.chdir(cwd)

        self._invoke_blocking(worker_id, target_module, target_name)

    def join(self):
        """Waits until all workers finish"""
        self._prune()
        while self.workers:
            time.sleep(0.05)
            self._prune()

    def archive_raw_inputs(self, archive_path: str):
        """Archives the raw data to the workers to the given path

        Args:
            archive_path (str): the path to archive data to
        """

        if not isinstance(archive_path, str):
            raise ValueError(f'expected archive path is str, got {archive_path}')
        self.join()

        working_path = _get_working_dir(self.identifier)
        zipdir(working_path)
        os.rename(working_path + '.zip', archive_path)
        self.workers_spawned = 0
        self._prepared = False

    def delete_raw_inputs(self):
        """Simply deletes the raw inputs instead of archiving them. Intended for
        when you are repeating the same command over and over."""

        cwd = os.getcwd()
        shutil.rmtree(_get_working_dir(self.identifier))
        os.chdir(cwd)

        self.workers_spawned = 0
        self._prepared = False
