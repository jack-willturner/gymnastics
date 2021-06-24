import torch
import torch.nn as nn


class MixConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        stride,
        bias,
        groups=2,
        padding=1,
    ):
        super(MixConv, self).__init__()

        assert len(kernel_sizes) == groups

        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        self.convs = []

        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Conv2d(
                    self.in_channels_per_group,
                    self.out_channels_per_group,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                )
            )

    def forward(self, x):
        outs = []
        for i, conv in enumerate(self.convs):
            outs.append(
                conv(
                    x[
                        :,
                        (i * self.in_channels_per_group) : (i + 1)
                        * self.in_channels_per_group,
                        :,
                        :,
                    ]
                )
            )
        return torch.cat(outs, 1)
