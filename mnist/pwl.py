"""Point with label producer for MNIST"""

import torch
import numpy as np
import struct

from shared.pwl import SimplePointWithLabelProducer
import os

class MNISTData:
    """This is the actual class that has the mnist dataset.
    Each value in this dataset is a tuple of inputs (as a
    tensor of pixels) and the corresponding label (byte 0-9)

    This dataset can be initialized from a byte array through
    the constructor or from the extracted file through the
    "load_from" class method.

    Attributes:
        raw_image_data (numpy.array):
            2-dimensional array. The first column corresponds with the
            image index, the second column corresponds with the pixel
            index. There are 24x24 pixels in the standard MNIST dataset,
            in row then column order.

        raw_label_data (numpy.array):
            1-dimensional array. The column corresponds with the image
            index and the value is a byte (0-9) that corresponds with
            the label.
    """

    def __init__(self, raw_image_data, raw_label_data):
        self.raw_image_data = raw_image_data
        self.raw_label_data = raw_label_data

    @classmethod
    def load_from(cls, image_filepath, labels_filepath):
        """Loads the MNIST data from the extracted files, assuming
        they are in the format described at http://yann.lecun.com/exdb/mnist/

        Args:
            image_filepath (str): The path to the image file
            labels_filepath (str): The path to the labels file

        Returns:
            data (MNistLabelledDataset): The loaded data set
        """

        img_data = None
        lab_data = None
        with open(image_filepath, 'rb') as handle:
            header = handle.read(16)
            magic_num, num_imgs, num_rows, num_cols = (
                struct.unpack('>4i', header)
            )

            assert magic_num == 2051

            pixels_per_image = num_rows*num_cols
            img_data = np.zeros((num_imgs, pixels_per_image), dtype='uint8')

            img_format = f'>{pixels_per_image}B'

            for img_index in range(num_imgs):
                img_data[img_index, :] = struct.unpack(
                    img_format, handle.read(pixels_per_image))

        with open(labels_filepath, 'rb') as handle:
            header = handle.read(8)
            magic_num, num_labs = (
                struct.unpack('>2i', header)
            )

            assert magic_num == 2049
            assert len(img_data) == num_labs

            lab_data = np.zeros(num_labs, dtype='uint8')
            num_remaining = num_labs
            cur_index = 0

            while num_remaining >= 4096:
                lab_data[cur_index:(cur_index+4096)] = (
                    struct.unpack('>4096B', handle.read(4096))
                )
                cur_index = cur_index + 4096
                num_remaining = num_remaining - 4096

            lab_data[cur_index:num_labs] = (
                struct.unpack(f'>{num_remaining}B', handle.read(num_remaining))
            )

        return cls(img_data, lab_data)

    @classmethod
    def load_train(cls):
        """Loads the training dataset"""
        return cls.load_from(os.path.join('data', 'mnist', 'train-images-idx3-ubyte'),
                             os.path.join('data', 'mnist', 'train-labels-idx1-ubyte'))

    @classmethod
    def load_test(cls):
        """Loads the test / validation dataset"""
        return cls.load_from(os.path.join('data', 'mnist', 't10k-images-idx3-ubyte'),
                             os.path.join('data', 'mnist', 't10k-labels-idx1-ubyte'))

    def __len__(self):
        return len(self.raw_label_data)

    def __getitem__(self, index):
        if index < 0 or index > len(self):
            return IndexError(str(index))
        return (torch.from_numpy(self.raw_image_data[index]),
                torch.tensor(self.raw_label_data[index], dtype=torch.uint8))

    def to_pwl(self) -> SimplePointWithLabelProducer:
        """Converts the given mnist data into a point with label producer. Does not
        perform any rescaling.
        """
        image_data = torch.from_numpy(self.raw_image_data).type(torch.double)
        label_data = torch.from_numpy(self.raw_label_data).type(torch.int32)
        return SimplePointWithLabelProducer(image_data, label_data, 10)

