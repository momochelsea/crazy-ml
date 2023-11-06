import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np  # linear algebra
import struct
from array import array
from os.path import join


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            st = i * rows * cols
            et = (i + 1) * rows * cols
            img = np.array(image_data[st:et])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


#
# MNIST Data Loader Class
#
class Loader:
    def __init__(self, input_path) -> None:
        self.device = torch.cuda.current_device()

        self.training_images_filepath = join(
            input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
        )
        self.training_labels_filepath = join(
            input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
        )
        self.test_images_filepath = join(
            input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
        )
        self.test_labels_filepath = join(
            input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
        )

    def get_ds(self):
        mnist_dataloader = MnistDataloader(
            self.training_images_filepath,
            self.training_labels_filepath,
            self.test_images_filepath,
            self.test_labels_filepath,
        )
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

        # 准备数据集

        # x 是 float 类型
        # y 是 int 类型

        x_train = torch.tensor(np.array(x_train, dtype=np.float32), dtype=torch.float32)
        x_test = torch.tensor(np.array(x_test, dtype=np.float32), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_train, dtype=np.int64), dtype=torch.int64)
        y_test = torch.tensor(np.array(y_test, dtype=np.int64), dtype=torch.int64)

        # print(x_train.shape, y_train.shape)
        # print(x_test.shape, y_test.shape)

        if torch.cuda.is_available():
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)

        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

    def get_dl(self, batch_size):
        train_ds, test_ds = self.get_ds()

        return DataLoader(train_ds, batch_size), DataLoader(test_ds, batch_size)
