import torch.nn as nn


class MaxPool2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(MaxPool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.maxpool(x)
