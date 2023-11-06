import torch
from torch import nn
from torch.nn import functional as F


class VerifyCode(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.cuda.current_device()

        # 3 * 30x30
        # => 256 * 30x30
        self.layer1 = nn.Conv2d(in_channels, 128, 5, padding=2, stride=1)
        # => 256 * 15x15
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # => 30 * 15x15
        self.layer2 = nn.Conv2d(128, 32, 3, padding=1, stride=1)
        # 30 * 7x7
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # 32 * 7x7 = 1568
        self.layer3 = nn.Linear(1568, 256)
        self.layer4 = nn.Linear(256, out_channels)

    def get_model(self):
        me = self
        if torch.cuda.is_available():
            me = me.to(self.device)

        return me

    def forward(self, x):
        l1 = self.layer1(x)
        l1r = F.relu(l1)
        p1 = self.pool1(l1r)
        # print("layer1", l1.shape, l1r.shape, p1.shape)

        l2 = self.layer2(p1)
        l2r = F.relu(l2)
        p2 = self.pool2(l2r)
        # print("layer2", l2.shape, l2r.shape, p2.shape)

        p2f = torch.flatten(p2, start_dim=1)
        # print("transform", p2f.shape)

        l3 = self.layer3(p2f)
        l3r = F.relu(l3)
        # print("layer3", l3.shape, l3r.shape)

        p3 = l3r
        # p3 = p2f

        return self.layer4(p3)
