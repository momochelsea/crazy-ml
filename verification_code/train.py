import sys

sys.path.append("..")

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from time import time


from dataset import Dataset
from model import VerifyCode
from utils import get_sec, get_ms, get_percent, num2char, char2num, show_img


def train(epochs, logic, optimizer, train_dl):
    stime = time()

    for epoch in range(epochs):
        print(f"== Epoch {epoch}")

        i = 0
        sstime = time()
        for xb, yb in train_dl:
            i += 1

            xb = torch.flatten(xb, start_dim=0, end_dim=1)
            yb = torch.flatten(yb, start_dim=0, end_dim=1)
            # print(type(xb), type(yb), xb.shape, yb.shape)

            xb = torch.unsqueeze(xb, 1)

            y_pred = logic(xb)
            # print(type(y_pred), y_pred.shape)

            loss = F.cross_entropy(y_pred, yb)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                cctime = get_ms(time(), sstime)
                print(f"i={i} loss: {loss} cost {cctime}ms")
                sstime = time()

    ctime = get_sec(time(), stime)
    print("===========================")
    print(f"loss: {loss}")
    print(f"cost: {ctime}s")


def test(logic, test_ds):
    x, y = test_ds[:]
    x = torch.flatten(x, start_dim=0, end_dim=1)
    y = torch.flatten(y, start_dim=0, end_dim=1)

    x = torch.unsqueeze(x, 1)

    y_pred = logic(x)
    # print(torch.argmax(y_pred[0:10], 1))
    # print(y[0:10])
    # print(torch.argmax(y[0:10]))

    accuracy = torch.argmax(y_pred, 1) == y
    accuracy = np.mean(accuracy.cpu().numpy())

    print(f"accuracy: {get_percent(accuracy)}%")

    return accuracy


# 读取数据集
DS_PATH = "../dataset/"
epochs = 200
batch_size = 32
learning_rate = 0.001


loader = Dataset(DS_PATH)
# loader.rewrite_ds()

train_ds, test_ds = loader.read_ds()

# x_train, y_train = train_ds[:]
# x_test, y_test = test_ds[:]
# print("ds shape: ", x_train.shape, y_train.shape)
# print(y_train[0])

# print(test_ds[0][0].shape, test_ds[0][1].shape)
# img = test_ds[0][0].cpu()
# code = test_ds[0][1].cpu()
# for i in range(img.shape[0]):
#     print(img[i].shape, num2char(code[i].item()))
#     show_img(img[i])

train_dl, _ = loader.read_dl(batch_size)

in_channels = 1
out_channels = 26 * 2 + 10

logic = VerifyCode(in_channels, out_channels).get_model()
optimizer = optim.SGD(logic.parameters(), lr=learning_rate)

train(epochs, logic, optimizer, train_dl)
test(logic, test_ds)

torch.save(logic, "../model/verification_code.pt")
