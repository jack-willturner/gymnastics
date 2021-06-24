import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)
