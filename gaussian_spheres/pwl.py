"""A gaussian spheres input technique. Randomly places cluster centers in an n-dimensional cube.
Then assigns each cluster a label. To generate points, a cluster is selected uniformly at random,
then a radius is selected from a normal distribution, then a point is selected uniformly from
within a sphere centered at the cluster center with the given radius."""

import torch
import typing
import scipy.spatial.distance as distance
import numpy as np

from shared.pwl import PointWithLabelProducer, PointWithLabel

class GaussianSpheresPWLP(PointWithLabelProducer):
    """Produces points selected from gaussian spheres. Marks are ignored.

    Attributes:
        clusters (list[PointWithLabel])
        radius_dist (torch.distributions.distribution.Distribution)
    """

    def __init__(self, epoch_size: int, input_dim: int, output_dim: int,
                 clusters: typing.List[PointWithLabel], std_dev: float, mean: float):
        super().__init__(epoch_size, input_dim, output_dim)
        self.clusters = clusters
        self.radius_dist = torch.distributions.normal.Normal(
            torch.tensor([float(mean)]), torch.tensor([float(std_dev)])) #pylint: disable=not-callable

    @classmethod
    def create(cls, epoch_size: int, input_dim: int, output_dim: int, cube_half_side_len: float,
               num_clusters: int, std_dev: float, mean: float, min_sep: float,
               force_split: bool = False):
        """Creates a new gaussian spheres pwlp, pulling points from the cube with a side length
        of 2*cube_half_side_len centered at the origin

        Arguments:
            epoch_size (int): the number of points we will consider 1 epoch
            input_dim (int): the input dimension (i.e., number of coordinates per point)
            output_dim (int): the output dimension (i.e., number of unique labels)
            cube_half_side_len (float): if '1', each coordinate is uniform from [-1, 1]
            num_clusters (int): the number of clusters
            std_dev (float): standard deviation of the radius
            mean (float): mean of the radius
            min_sep (float): minimum separation between points
            force_split (bool, optional): if True then there will be an even as possible
                distribution of cluster labels. if False then there will be a multinomial
                distribution of cluster labels with the same probability for each
        """
        # rejection sampling

        clust_centers = np.zeros((num_clusters, input_dim), dtype='double')

        clusters = []
        if force_split:
            next_label = 0
        for i in range(num_clusters):
            rejections = 0

            center = torch.zeros((input_dim,), dtype=torch.double)

            while True:
                torch.rand(input_dim, out=center)
                center = (center - 0.5) * 2 * cube_half_side_len

                distances = distance.cdist(center.reshape(1, -1).numpy(), clust_centers)
                if np.min(distances) < min_sep:
                    rejections += 1
                    if rejections > 10000:
                        raise ValueError('rejected too many points!')
                else:
                    break

            clust_centers[i, :] = center.numpy()
            if force_split:
                clust_label = next_label
                next_label = (next_label + 1) % output_dim
            else:
                clust_label = torch.randint(output_dim, (1,)).item()
            clusters.append(PointWithLabel(point=center, label=clust_label))

        return cls(epoch_size, input_dim, output_dim, clusters, std_dev, mean)

    def _fill(self, points: torch.tensor, labels: torch.tensor):
        batch_size = points.shape[0]

        cluster_inds = torch.randint(len(self.clusters), (batch_size,), dtype=torch.long)
        vec = torch.zeros((self.input_dim,), dtype=torch.double)

        for i in range(batch_size):
            clust = self.clusters[cluster_inds[i].item()]
            radius = torch.abs(self.radius_dist.sample()).double()

            torch.randn(self.input_dim, out=vec)
            vec *= (radius / torch.norm(vec))
            labels[i] = clust.label
            points[i, :] = clust.point + vec

    def _position(self, pos: int):
        pass
