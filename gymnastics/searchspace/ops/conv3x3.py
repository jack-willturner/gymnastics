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
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)
