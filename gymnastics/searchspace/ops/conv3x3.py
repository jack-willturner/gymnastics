import torch.nn as nn


class Conv3x3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=None,
        stride=1,
        bias=False,
        padding=1,
    ):
        super(Conv3x3, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        self.conv.in_channels = x.size()

        return self.conv(x)

    def __str__(self):
        return f"Conv3x3({self.conv.in_channels} -> {self.conv.out_channels})"

    def __repr__(self):
        return f"Conv3x3({self.conv.in_channels} -> {self.conv.out_channels})"
