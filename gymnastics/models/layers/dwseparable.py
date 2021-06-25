import torch
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
        """Depthwise-separable convolution (i.e. grouped convolution with groups=in_channels=out_channels).
        If in_channels != out_channels, then any remaining out_channels will just be computed normally.
        """
        super(DWSeparableConv, self).__init__()

        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            bias=bias,
            padding=padding,
        )

        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        out = self.dw_conv(x)
        return self.pw_conv(out)
