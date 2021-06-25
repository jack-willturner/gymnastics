import torch
import torch.nn as nn


class Zeroize(nn.Module):
    def __init__(self):
        super(self, Zeroize).__init__()

    def forward(self, x):
        return torch.zeros(x.size())
