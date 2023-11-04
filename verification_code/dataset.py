import os
import matplotlib.image as mpimg
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# '0-9' 48-57 => -48 => 0-9
# 'A-Z' 65-90 => -55 => 10-35
# 'a-z' 97-122 => -61 => 36-61


def num2char(num) -> str:
    if num >= 0 and num <= 9:
        num += 48
        return chr(num)
    if num >= 10 and num <= 35:
        num += 55
        return chr(num)
    if num >= 36 and num <= 61:
        num += 61
        return chr(num)
    return ""


def char2num(char) -> int:
    num = ord(char)
    # print(char, num)

    if num >= 48 and num <= 57:
        num -= 48
        return num
    if num >= 65 and num <= 90:
        num -= 55
        return num
    if num >= 97 and num <= 122:
        num -= 61
        return num
    return -1


class Dataset:
    def __init__(self, ds_path):
        self.ds_path = ds_path
        self.paths = [ds_path + name for name in os.listdir(ds_path)]

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

        np.savez(
            self.ds_path + "verification_code.npz",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def read_ds(self):
        data = np.load(self.ds_path + "verification_code.npz")
        # print(data["x_train"].shape, data["y_train"].shape)
        # print(data["x_test"].shape, data["y_test"].shape)
        # print(type(data["x_train"]), type(data["y_train"]))
        # print(type(data["x_test"]), type(data["y_test"]))

        x_train = torch.tensor(data["x_train"], dtype=torch.float32)
        y_train = torch.tensor(data["y_train"], dtype=torch.int64)
        x_test = torch.tensor(data["x_test"], dtype=torch.float32)
        y_test = torch.tensor(data["y_test"], dtype=torch.int64)

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
                np.split(np.array(mpimg.imread(path)), 5, axis=1)
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
    DATESET_PATH = "../dataset/verification_code/"

    loader = Dataset(DATESET_PATH)
    # loader.rewrite_ds()

    train_ds, test_ds = loader.read_ds()
    print(train_ds, test_ds)
    print(type(train_ds), type(test_ds))

    train_dl, test_dl = loader.read_dl(32)
    print(train_dl, test_dl)
    print(type(train_dl), type(test_dl))
