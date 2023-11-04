import os
from PIL import Image
import matplotlib.image as mpimg
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class VerifyCode(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 3 * 30x30
        # => 256 * 30x30
        self.layer1 = nn.Conv2d(in_channels, 16, 5, padding=2, stride=1)
        # => 256 * 15x15
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # => 30 * 15x15
        self.layer2 = nn.Conv2d(16, 8, 3, padding=1, stride=1)
        # 30 * 7x7
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # 30 * 7x7 = 1470
        # self.layer3 = nn.Linear(392, 256)
        self.layer4 = nn.Linear(392, out_channels)

    def forward(self, x):
        l1 = self.layer1(x)
        l1r = F.softmax(l1, dim=1)
        p1 = self.pool1(l1r)
        # print("layer1", l1.shape, l1r.shape, p1.shape)

        l2 = self.layer2(p1)
        l2r = F.softmax(l2, dim=1)
        p2 = self.pool2(l2r)
        # print("layer2", l2.shape, l2r.shape, p2.shape)

        p2f = torch.flatten(p2, start_dim=1)
        # print("transform", p2f.shape)

        # l3 = self.layer3(p2f)
        # l3r = F.relu(l3)
        # print("layer3", l3.shape, l3r.shape)

        p3 = p2f
        p3 = p2f

        return self.layer4(p3)

    def test(self, x, y):
        pass
