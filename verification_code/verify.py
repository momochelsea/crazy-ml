import sys

sys.path.append("..")

import torch

from dataset import Dataset
from utils import num2char


DS_PATH = "../dataset/"
loader = Dataset(DS_PATH)
img = loader.read_img("2e5p1.jpg")
img = torch.flatten(img, start_dim=0, end_dim=1)
img = torch.unsqueeze(img, 1)
print(img.shape)

logic = torch.load("../model/verification_code.pt")

code = logic(img)
code = torch.argmax(code, dim=1)
print(code)
code = list(map(num2char, code))
print(code)
