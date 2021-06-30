import torch.nn as nn
from .conv1x1 import Conv1x1


class AvgPool2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(AvgPool2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            self.projection = Conv1x1(in_channels, out_channels)
        else:
            self.projection = nn.Identity()

        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.avgpool(self.projection(x))

    def __str__(self):
        return f"AvgPool2d({self.avgpool.kernel_size}, stride={self.avgpool.stride}, padding={self.avgpool.padding})"

    def __repr__(self):
        return f"AvgPool2d({self.avgpool.kernel_size}, stride={self.avgpool.stride}, padding={self.avgpool.padding})"
