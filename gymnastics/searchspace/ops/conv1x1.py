import torch.nn as nn


class Conv1x1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=None,
        stride=1,
        bias=False,
        padding=0,
    ):
        super(Conv1x1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)

    def __str__(self):
        return f"Conv1x1({self.conv.in_channels} -> {self.conv.out_channels})"

    def __repr__(self):
        return f"Conv1x1({self.conv.in_channels} -> {self.conv.out_channels})"
