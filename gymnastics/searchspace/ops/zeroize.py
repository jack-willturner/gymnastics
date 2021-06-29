import torch
import torch.nn as nn


class Zeroize(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Zeroize, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 0

    def forward(self, x):
        return torch.zeros(x.size())

    def __str__(self):
        return f"Zeroize"

    def __repr__(self):
        return f"Zeroize"
