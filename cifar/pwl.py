"""Handles loading the CIFAR dataset"""

import os
import numpy as np
import torch
import torchvision # pylint: disable=import-error

from shared.pwl import SimplePointWithLabelProducer

DATA_FOLDER = os.path.join('data', 'cifar')

class CIFARData:
    """Describes the cifar data as loaded from file. These use numpy arrays

    Attributes:
        data (np array of num_imgsx32x32x3 numpy arrays of uint8s) in HxWxC format
        labels (np array of num_imgs uint8 labels)
        classes (list): the index corresponds to the label number, the value to label class
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, classes: list):
        if not isinstance(data, np.ndarray):
            raise ValueError(f'expected data is ndarray, got {data} (type={type(data)})')
        if len(data.shape) != 4:
            raise ValueError(f'expected data.shape is [num_imgs, 32, 32, 3] got {data.shape}')
        if data.dtype != np.uint8:
            raise ValueError(f'expected data.dtype is uint8, got {data.dtype}')
        if data.shape[1] != 32 or data.shape[2] != 32 or data.shape[3] != 3:
            raise ValueError(f'expected data.shape is [num_imgs, 32, 32, 3], got {data.shape}')
        if not isinstance(labels, np.ndarray):
            raise ValueError(f'expected labels is ndarray, got {labels} (type={type(labels)})')
        if len(labels.shape) != 1:
            raise ValueError(f'expected labels.shape is [num_imgs], got {labels.shape}')
        if labels.dtype != np.uint8:
            raise ValueError(f'expected labels.dtype is uint8, got {labels.dtype}')
        if labels.shape[0] != data.shape[0]:
            raise ValueError(f'expected data.shape={data.shape} and labels.shape={labels.shape} match on dim 0')
        if not isinstance(classes, list):
            raise ValueError(f'expected classes is list, got {classes} (type={type(classes)})')
        for idx, val in enumerate(classes):
            if not isinstance(val, str):
                raise ValueError(f'expected classes[{idx}] is str, got {val} (type={type(val)})')
        self.data = data
        self.labels = labels
        self.classes = classes

    @classmethod
    def load_train(cls):
        """Downloads (if necessary) and loads the train data"""
        data = torchvision.datasets.CIFAR10(DATA_FOLDER, train=True, download=True)
        import pdb; pdb.set_trace()
        return cls(data.data, np.array(data.targets, dtype='uint8'), data.classes)

    @classmethod
    def load_test(cls):
        """Downloads (if necessary) and loads the test data"""
        data = torchvision.datasets.CIFAR10(DATA_FOLDER, train=False, download=True)
        return cls(data.data, np.array(data.targets, dtype='uint8'), data.classes)

    def to_pwl(self):
        """Converts this to a point-with-label producer, which requires
        reshaping to CxHxW and flattening"""
        return SimplePointWithLabelProducer(
            torch.from_numpy(self.data.transpose((0, 3, 1, 2)).view(-1, 32*32*3)).double(),
            torch.from_numpy(self.labels).long(),
            len(self.classes)
        )
