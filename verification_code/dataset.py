import sys

sys.path.append("..")

import os
import matplotlib.image as mpimg
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils import char2num


class Dataset:
    def __init__(self, npz_path):
        self.device = torch.cuda.current_device()

        self.npz_path = npz_path
        self.ds_path = npz_path + "verification_code/"
        self.paths = [self.ds_path + name for name in os.listdir(self.ds_path)]

    def read_img(self, name):
        img = self._get_x([self.ds_path + name])
        img = torch.tensor(img, dtype=torch.float32, device=self.device)

        return img

    def rewrite_ds(self):
        # random for test dataset
        tests = np.random.randint(0, len(self.paths), 2000)

        # pick paths for train and test
        train_paths = [self.paths[i] for i in range(len(self.paths)) if i not in tests]
        test_paths = [self.paths[i] for i in range(len(self.paths)) if i in tests]

        # get dataset for train and test
        x_train = self._get_x(train_paths)
        y_train = self._get_y(train_paths)
        x_test = self._get_x(test_paths)
        y_test = self._get_y(test_paths)

        # print(x_train.shape, y_train.shape)
        # print(x_test.shape, y_test.shape)

        # x_train = torch.tensor(x_train, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.int64)
        # x_test = torch.tensor(x_test, dtype=torch.float32)
        # y_test = torch.tensor(y_test, dtype=torch.int64)

        np.savez(
            self.npz_path + "verification_code.npz",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def read_ds(self):
        data = np.load(self.npz_path + "verification_code.npz")
        # print(data["x_train"].shape, data["y_train"].shape)
        # print(data["x_test"].shape, data["y_test"].shape)
        # print(type(data["x_train"]), type(data["y_train"]))
        # print(type(data["x_test"]), type(data["y_test"]))

        x_train = torch.tensor(data["x_train"], dtype=torch.float32)
        y_train = torch.tensor(data["y_train"], dtype=torch.int64)
        x_test = torch.tensor(data["x_test"], dtype=torch.float32)
        y_test = torch.tensor(data["y_test"], dtype=torch.int64)

        if torch.cuda.is_available():
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)

        # print(x_train.shape, y_train.shape)
        # print(x_test.shape, y_test.shape)
        # print(type(x_train), type(y_train))
        # print(type(x_test), type(y_test))

        # return (x_train, y_train), (x_test, y_test)
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

    def read_dl(self, batch_size):
        train, test = self.read_ds()

        return DataLoader(train, batch_size), DataLoader(test, batch_size)

    def _get_x(self, paths):
        return np.array(
            [
                np.split(np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE)), 5, axis=1)
                for path in paths
                # if path[-9:] == "00175.jpg"
                if path[-3:] == "jpg"
            ]
        )

    def _get_y(self, paths):
        return np.array(
            [
                np.array([char2num(char) for char in [*path[-9:-4]]])
                for path in paths
                # if path[-9:] == "00175.jpg"
                if path[-3:] == "jpg"
            ]
        )


if __name__ == "__main__":
    DS_PATH = "../dataset/"

    loader = Dataset(DS_PATH)
    loader.rewrite_ds()

    train_ds, test_ds = loader.read_ds()
    print(train_ds, test_ds)
    print(type(train_ds), type(test_ds))

    train_dl, test_dl = loader.read_dl(32)
    print(train_dl, test_dl)
    print(type(train_dl), type(test_dl))
