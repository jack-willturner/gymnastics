import torch.nn as nn


class GConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=False,
        groups=1,
        padding=1,
    ):
        super(GConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)

    def __str__(self):
        return f"GConv{self.conv.kernel_size}x{self.conv.kernel_size}"

    def __repr__(self):
        return f"GConv{self.conv.kernel_size}x{self.conv.kernel_size}"
