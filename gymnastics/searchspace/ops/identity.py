import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Identity, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

    def forward(self, x):
        return x

    def __str__(self):
        return f"Identity"

    def __repr__(self):
        return f"Identity"
