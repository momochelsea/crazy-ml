import sys

sys.path.append("..")

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch import nn
from torch.nn import functional as F

import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from utils import show_img

# model = torch.load("../model/mnist_fc.pt")
# model = torch.load("../model/mnist_lenet.pt")
model = torch.load("../model/mnist_vgg.pt")

for i in range(10):
    filename = "../dataset/mnist/tests/" + str(i) + ".jpg"
    file = torch.tensor(
        np.array(cv2.imread(filename, cv2.IMREAD_GRAYSCALE)), dtype=torch.float32
    )
    # print(file.shape)
    # show_img(file)
    # break

    # FC
    # file = file.reshape(1, file.shape[0] * file.shape[1])
    # LeNet, VGG
    file = torch.reshape(file, (1, 1, file.shape[0], file.shape[1]))

    if torch.cuda.is_available():
        file = file.to(torch.cuda.current_device())

    y_pred = model(file)
    # print(y_pred)
    print(i, "==", torch.argmax(y_pred, 1).item())
