import torch.nn as nn


class DWSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias,
        groups=1,
        padding=1,
    ):
        super(DWSeparableConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)
