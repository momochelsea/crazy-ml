import sys

sys.path.append("..")

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from utils import get_percent

import dataset
import module


def train(epochs, module, optimizer, train_dl):
    for epoch in range(epochs):
        print(f"== Epoch {epoch+1}")

        i = 0
        for xb, yb in train_dl:
            i += 1

            # 处理数据为一维
            xb = xb.reshape(xb.shape[0], xb.shape[1] * xb.shape[2])
            # print(train_x.shape)

            y_pred = module(xb)

            loss = F.cross_entropy(y_pred, yb)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print(f"{i} loss: {loss}")

        print(f"\nEpoch {epoch+1}: loss: {loss}\n")
        torch.save(module, "../model/mnist_fc.pt")


def test(model, test_ds):
    x, y = test_ds[:]

    # 处理数据为一维
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    # print(train_x.shape)

    y_pred = model(x)
    # print(torch.argmax(y_pred[0:10], 1))
    # print(y[0:10])
    # print(torch.argmax(y[0:10]))

    accuracy = torch.argmax(y_pred, 1) == y
    accuracy = np.mean(accuracy.cpu().numpy())

    print(f"accuracy: {get_percent(accuracy)}%")

    return accuracy


input_path = "../dataset/mnist/"
epochs = 100
bs = 64
lr = 0.001

loader = dataset.Loader(input_path=input_path)
train_ds, test_ds = loader.get_ds()
train_dl, test_dl = loader.get_dl(bs)

print("ds shape", train_ds[0][0].shape, train_ds[0][1].shape)

module = module.FC(784, 10).get_model()
optimizer = optim.SGD(module.parameters(), lr=lr)

train(epochs, module, optimizer, train_dl)
test(module, test_ds)
