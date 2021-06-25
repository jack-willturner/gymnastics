import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(self, Identity).__init__()

    def forward(self, x):
        return x
