import torch
import torch.nn as nn


class ShiftConv(nn.Module):
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
        super(ShiftConv, self).__init__()

        # each group takes a different shift direction
        groups = kernel_size ** 2

        self.missing_out_channels = out_channels % groups

        in_channels = in_channels - (in_channels % groups)
        out_channels = out_channels - (out_channels % groups)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            bias=bias,
            padding=padding,
        )

        # initialise each shift group
        channels_per_group = out_channels // groups

        indices_of_on_pixel = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                indices_of_on_pixel.append((i, j))

        for group in range(groups):
            i, j = indices_of_on_pixel[group]

            for channel in range(channels_per_group):

                shift_weight = torch.zeros(kernel_size, kernel_size)

                shift_weight[i, j] = 1.0

                self.conv.weight.data[
                    (group * channels_per_group) + channel
                ] = shift_weight

        self.pointwise = nn.Conv2d(
            out_channels + self.missing_out_channels,
            out_channels + self.missing_out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):

        with torch.no_grad():
            out = self.conv(x[:, : self.conv.in_channels, :, :])

            if self.missing_out_channels > 0:
                # grab the central group and repeat it
                centre_shift = out[:, self.conv.out_channels // 2, :, :]

                centre_shift = centre_shift.unsqueeze(1).repeat(
                    1, self.missing_out_channels, 1, 1
                )

                out = torch.cat([out, centre_shift], 1)

        return self.pointwise(out)
