import torch
from torch import nn
from torch.nn import functional as F


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


class CNN(nn.Module):
    def __init__(self, inputs, outputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.cuda.current_device()

        self.layer1 = nn.Conv2d(
            inputs, 256, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(256, 64, kernel_size=2, padding=1, stride=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, outputs, bias=True)

    def get_model(self):
        me = self
        if torch.cuda.is_available():
            me = me.to(self.device)

        return me

    def forward(self, x):
        t1 = self.pool1(F.relu(self.layer1(x)))
        t2 = self.pool2(F.relu(self.layer2(t1)))
        tt = torch.flatten(t2, start_dim=1)

        return self.fc1(tt)


class VGG(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.cuda.current_device()

    def get_model(self):
        me = self
        if torch.cuda.is_available():
            me = me.to(self.device)

        return me

    def forward(self, x):
        pass
