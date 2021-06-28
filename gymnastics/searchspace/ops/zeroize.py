import torch
import torch.nn as nn


class Zeroize(nn.Module):
    def __init__(self, in_planes, out_planes, **kwargs):
        super(Zeroize, self).__init__()

    def forward(self, x):
        return torch.zeros(x.size())

    def __str__(self):
        return f"Zeroize"

    def __repr__(self):
        return f"Zeroize"
