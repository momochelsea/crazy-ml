import torch
from torch import nn
from torch.nn import functional as F
import math


class FC(nn.Module):
    def __init__(self, inputs, outputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.cuda.current_device()

        self.layer1 = nn.Linear(inputs, 96)
        self.layer2 = nn.Linear(96, outputs)

    def get_model(self):
        me = self
        if torch.cuda.is_available():
            me = me.to(self.device)

        return me

    def forward(self, x):
        t1 = self.layer1(x)
        t1r = F.relu(t1)
        return self.layer2(t1r)


class LeNet(nn.Module):
    def __init__(self, inputs, outputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.cuda.current_device()
        # 1x28*28 => 6@1x5*5 => 6x28*28
        self.layer1 = nn.Conv2d(
            inputs, 6, kernel_size=5, padding=2, stride=1, bias=True
        )
        # 6x28*28 / 2 => 6x14*14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 6x14*14 => 16@6x5*5 => 16x14*14
        self.layer2 = nn.Conv2d(6, 16, kernel_size=5, padding=2, stride=1, bias=True)
        # 16x14*14 / 2 => 16x7*7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 7 * 7, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, outputs, bias=True)

    def get_model(self):
        me = self
        if torch.cuda.is_available():
            me = me.to(self.device)

        return me

    def forward(self, x):
        t1 = self.pool1(F.relu(self.layer1(x)))
        t2 = self.pool2(F.relu(self.layer2(t1)))
        tt = torch.flatten(t2, start_dim=1)
        f1 = self.fc1(tt)
        f2 = self.fc2(f1)
        f3 = self.fc3(f2)

        return f3


class VGG(nn.Module):
    def __init__(self, width, archs, inputs, outputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.cuda.current_device()
        self.seq = self.get_layers(width, archs, inputs, outputs)

    def get_blocks(self, convs, inputs, outputs):
        layers = []

        for _ in range(convs):
            layer = nn.Conv2d(
                in_channels=inputs,
                out_channels=outputs,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            layers.append(layer)
            layers.append(nn.ReLU())
            inputs = outputs

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def get_layers(self, width, archs, inputs, outputs):
        blocks = []

        ipt = inputs
        blk = len(archs)

        for i in range(blk):
            convs, opt = archs[i]
            block = self.get_blocks(convs, ipt, opt)
            blocks.append(block)
            ipt = opt

        width = math.floor(width / (2**blk))
        return nn.Sequential(
            *blocks,
            nn.Flatten(start_dim=1),
            nn.Linear(ipt * width * width, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, outputs)
        )

    def get_model(self):
        me = self
        if torch.cuda.is_available():
            me = me.to(self.device)

        return me

    def forward(self, x):
        return self.seq(x)
