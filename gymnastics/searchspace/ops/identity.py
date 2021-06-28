import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, in_planes, out_planes, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def __str__(self):
        return f"Identity"

    def __repr__(self):
        return f"Identity"
