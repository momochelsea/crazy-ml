import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

from dataset import Dataset
from model import VerifyCode


def train(epochs, logic, optimizer, train_dl):
    for epoch in range(epochs):
        print(f"== Epoch {epoch}")

        i = 0
        for xb, yb in train_dl:
            i += 1

            xb = torch.flatten(xb, start_dim=0, end_dim=1)
            yb = torch.flatten(yb, start_dim=0, end_dim=1)
            # print(type(xb), type(yb), xb.shape, yb.shape)

            xb = torch.swapdims(xb, 1, 3)

            y_pred = logic(xb)
            # print(type(y_pred), y_pred.shape)

            loss = F.cross_entropy(y_pred, yb)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % 20 == 0:
                print(f"i={i} loss: {loss}")

        print(f"\nloss: {loss}\n")


def test(logic, test_dl):
    tp = np.zeros(62)
    tn = np.zeros(62)
    fp = np.zeros(62)
    fn = np.zeros(62)


# 读取数据集
DATESET_PATH = "../dataset/verification_code/"
epochs = 5
batch_size = 32
learning_rate = 0.001


loader = Dataset(DATESET_PATH)
# loader.rewrite_ds()

# train_ds, test_ds = loader.read_ds()
# print(train_ds, test_ds)
# print(type(train_ds), type(test_ds))

# x_train, y_train = train_ds[:]
# x_test, y_test = test_ds[:]

# print(type(x_train), x_train.shape)
# print(type(y_train), y_train.shape)
# print(y_train[0])

train_dl, test_dl = loader.read_dl(batch_size)
print(train_dl, test_dl)
print(type(train_dl), type(test_dl))

in_channels = 3
out_channels = 26 * 2 + 10

logic = VerifyCode(in_channels, out_channels)
optimizer = optim.SGD(logic.parameters(), lr=learning_rate)

train(epochs, logic, optimizer, train_dl)
