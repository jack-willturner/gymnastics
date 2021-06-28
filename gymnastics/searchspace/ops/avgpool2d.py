import torch.nn as nn


class AvgPool2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(AvgPool2d, self).__init__()

        # register that channels won't change
        self.in_planes = in_planes
        self.out_planes = in_planes

        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.avgpool(x)

    def __str__(self):
        return f"AvgPool2d({self.avgpool.kernel_size})"

    def __repr__(self):
        return f"AvgPool2d({self.avgpool.kernel_size})"
