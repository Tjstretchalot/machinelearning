"""Not exactly a measure in itself, however this module is meant for detecting
clusters that occur within pc-space. It is based of scikits optics clustering.
Much of this is generic clustering and does not benefit from being specific to
pcs and thus a more general naming scheme is adopted
"""
import typing
import os
import json
import numpy as np
import sklearn.cluster
import sklearn.metrics

if not hasattr(sklearn.cluster, 'OPTICS'):
    raise ValueError(f'update scikit-learn to >=0.21.2 for clustering!')

import shared.typeutils as tus
import shared.filetools as filetools
import shared.measures.utils as mutils

class Clusters:
    """The data class which stores information about clusters generated from
    particular samples.

    Attributes:
        samples (ndarray[n_samples, n_features]): the samples that the clusters were
            selected from.
        centers (ndarray[n_clusters, n_features]): where the cluster centers are located
        labels (ndarray[n_samples]): each value is 0,1,...,n_clusters-1 and corresponds
            to the nearest cluster to the corresponding sample in pc-space

        calculate_params (dict[str, any]): the parameters that were used to generate these
            clusters.
    """

    def __init__(self, samples: np.ndarray, centers: np.ndarray, labels: np.ndarray,
                 calculate_params: typing.Dict[str, typing.Any]):
        tus.check(samples=(samples, np.ndarray), centers=(centers, np.ndarray),
                  labels=(labels, np.ndarray), calculate_params=(calculate_params, dict))
        tus.check_tensors(
            samples=(samples, ('n_samples', 'n_features'),
                     (np.dtype('float32'), np.dtype('float64'))),
            centers=(
                centers,
                ('n_clusters',
                 ('n_features', samples.shape[1] if len(samples.shape) > 1 else None)
                ),
                samples.dtype
            ),
            labels=(
                labels,
                (('n_samples', samples.shape[0] if bool(samples.shape) else None),),
                (np.dtype('int32'), np.dtype('int64'))
            )
        )
        self.samples = samples
        self.centers = centers
        self.labels = labels
        self.calculate_params = calculate_params

        self._bounds = None

    @property
    def num_samples(self):
        """Returns the number of samples used to generate these clusters"""
        return self.samples.shape[0]

    @property
    def num_features(self):
        """Returns the number of features in the sample space"""
        return self.samples.shape[1]

    @property
    def num_clusters(self):
        """Returns the number of clusters found. This may have been chosen"""
        return self.centers.shape[0]

    def get_bounds(self, cluster_ind: int) -> np.ndarray:
        """Gets the minimum/maximum bounds of the given cluster. This is
        the tightest box that completely contains all the points inside the cluster,
        one per feature. The result is cached.

        Arguments:
            cluster_ind (int): which cluster you are interested in the bounds in

        Returns:
            min_bounds (ndarray[num_features]): the min for each feature
            max_bounds (ndarray[num_features]): the max for each feature
        """
        if self._bounds is not None and cluster_ind in self._bounds:
            return self._bounds[cluster_ind]

        self._bounds = self._bounds or dict()

        min_features = self.samples[0].copy()
        max_features = min_features.copy()

        for sample in self.samples:
            np.minimum(min_features, sample, out=min_features)
            np.maximum(max_features, sample, out=max_features)

        self._bounds[cluster_ind] = (min_features, max_features)
        return min_features, max_features

    def save(self, filepath: str, exist_ok: bool = False, compress: bool = True) -> None:
        """Saves these clusters along with a description about how to load them
        to the given filepath. If the filepath has an extension, it must be .zip
        and it will be ignored in favor of compress.

        Arguments:
            filepath (str): the folder or zip file where these clusters should be
                saves
            exist_ok (bool): effects the behavior if the folder or zip file already
                exists. If this is False, then an error is thrown. If this is True,
                the existing files are deleted
            compress (bool): if True, the folder is compressed to a zip file after
                saving and the folder is deleted. If False, the result is left as a
                folder
        """

        outfile, outfile_wo_ext = mutils.process_outfile(filepath, exist_ok, compress)

        if os.path.exists(outfile_wo_ext):
            filetools.deldir(outfile_wo_ext)

        os.makedirs(outfile_wo_ext)

        np.savez_compressed(
            os.path.join(outfile_wo_ext, 'clusters.npz'),
            samples=self.samples,
            centers=self.centers,
            labels=self.labels
        )

        with open(os.path.join(outfile_wo_ext, 'calculate_params.json'), 'w') as out:
            json.dump(self.calculate_params, out)

        with open(os.path.join(outfile_wo_ext, 'readme.md'), 'w') as out:
            def _print(*args, **kwargs):
                print(*args, **kwargs, file=out)

            _print('Clusters')
            _print('  clusters.npz:')
            _print('    samples [n_samples, n_features] - the samples the clusters were calculated'
                   + ' from')
            _print('    centers [n_clusters, n_features] - the centers of the clusters')
            _print('    labels [n_samples] - the index in centers for the closest cluster '
                   + 'to each label')
            _print('  calculate_params.json:')
            _print('    Varies. Gives information about how clusters were calculated')

        if compress:
            if os.path.exists(outfile):
                os.remove(outfile)
            filetools.zipdir(outfile_wo_ext)

    @classmethod
    def load(cls, filepath: str, compress: bool = True):
        """Loads the clusters located in the given filepath. If the filepath has
        an extension it must be .zip and it will be ignored. This will first check
        if the folder exists and then the archive.

        Arguments:
            filepath (str): the path to the folder or archive that the clusters were saved in
            compress (bool): if True the folder will be compressed after this is done,
                regardless of the old state. If this is False, the folder will not be
                compressed after this is done, regardless of the old state.
        """

        outfile, outfile_wo_ext = mutils.process_outfile(filepath, True, False)

        if not os.path.exists(outfile_wo_ext):
            if not os.path.exists(outfile):
                raise FileNotFoundError(filepath)
            filetools.unzip(outfile)

        try:
            clusters_path = os.path.join(outfile_wo_ext, 'clusters.npz')
            if not os.path.exists(clusters_path):
                raise FileNotFoundError(clusters_path)

            calc_params_path = os.path.join(outfile_wo_ext, 'calculate_params.json')
            if not os.path.exists(calc_params_path):
                raise FileNotFoundError(calc_params_path)

            with np.load(clusters_path) as clusters:
                samples = clusters['samples']
                centers = clusters['centers']
                labels = clusters['labels']

            with open(calc_params_path, 'r') as infile:
                calculate_params = json.load(infile)

            return Clusters(samples, centers, labels, calculate_params)
        finally:
            if compress and os.path.exists(outfile_wo_ext):
                filetools.zipdir(outfile_wo_ext)

def find_clusters(samples: np.ndarray) -> Clusters:
    """Attempts to locate clusters in the given samples in the most generic
    way possible."""
    args = {
        'min_samples': 5,
        'max_eps': np.inf,
        'metric': 'precomputed',
        'p': 2,
        'metric_params': None,
        'cluster_method': 'xi',
        'eps': None,
        'xi': 0.05,
        'predecessor_correction': True,
        'min_cluster_size': 0.2,
        'algorithm': 'auto',
        'leaf_size': 30,
        'n_jobs': None
    }
    args_meta = {
        'precompute_metric': 'minkowski',
        'method': 'sklearn.cluster.OPTICS',
        'nearest_center_metric': 'euclidean' # this is for associating points AFTER clusters found
    }

    # compute clusters
    precomp = sklearn.metrics.pairwise_distances(samples, metric=args_meta['precompute_metric'])
    optics = sklearn.cluster.OPTICS(**args)
    optics.fit(precomp)

    # optics makes a heirarchy, but we wish to flatten that. first we determine
    # how many clusters there are which actually have points belonging to them
    # -1 is for unclustered points, which we will clsuter later
    labels = optics.labels_
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        unique_labels = np.ascontiguousarray(unique_labels[unique_labels != -1])

    # we are also going to want to centers of our labels
    sums = np.zeros((unique_labels.shape[0], samples.shape[1]), dtype='float64')
    num_per = np.zeros(unique_labels.shape[0], dtype='int64')
    new_labels = np.zeros(samples.shape[0], dtype='int32')

    # crunch numbers
    for lbl_ind, lbl in enumerate(unique_labels):
        mask = labels == lbl
        new_labels[mask] = lbl
        masked = samples[mask]
        sums[lbl_ind] = masked.sum(axis=0)
        num_per[lbl_ind] = masked.shape[0]

    if unique_labels.shape[0] == 1 and num_per[0] == samples.shape[0]:
        return Clusters(
            samples,
            np.zeros((0, samples.shape[1]), dtype='float32'),
            np.zeros((samples.shape[0],), dtype='int32'),
            {'clustering': args, 'other': args_meta}
        )

    # calculate centers of each cluster
    centers = (
        sums / (
            num_per.astype('float64')
            .reshape(num_per.shape[0], 1)
            .repeat(sums.shape[1], 1)
        )
    ).astype(samples.dtype)

    return Clusters(samples, centers, new_labels, {
        'clustering': args,
        'other': args_meta
    })
