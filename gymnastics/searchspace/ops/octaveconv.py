"""
https://github.com/vivym/OctaveConv.pytorch/blob/master/models/octave_resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class OctConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        alpha_in=0.25,
        alpha_out=0.25,
        type="normal",
        bias=False,
    ):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        if type == "first":
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)
            self.convh = nn.Conv2d(
                in_channels,
                hf_ch_out,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            )
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.convl = nn.Conv2d(
                in_channels,
                lf_ch_out,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            )
        elif type == "last":
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)
            self.convh = nn.Conv2d(
                hf_ch_in, out_channels, kernel_size=kernel_size, padding=padding
            )
            self.convl = nn.Conv2d(
                lf_ch_in, out_channels, kernel_size=kernel_size, padding=padding
            )
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
        else:
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)

            self.L2L = nn.Conv2d(
                lf_ch_in, lf_ch_out, kernel_size=kernel_size, stride=1, padding=padding
            )
            self.L2H = nn.Conv2d(
                lf_ch_in, hf_ch_out, kernel_size=kernel_size, stride=1, padding=padding
            )
            self.H2L = nn.Conv2d(
                hf_ch_in, lf_ch_out, kernel_size=kernel_size, stride=1, padding=padding
            )
            self.H2H = nn.Conv2d(
                hf_ch_in, hf_ch_out, kernel_size=kernel_size, stride=1, padding=padding
            )
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
            self.avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)

    def forward(self, x):
        if self.type == "first":
            if self.stride == 2:
                x = self.downsample(x)

            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)

            return hf, lf
        elif self.type == "last":
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.convh(hf) + self.convl(lf)
            else:
                return self.convh(hf) + self.convl(self.upsample(lf))
        else:
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.H2H(hf) + self.L2H(lf), self.L2L(
                    F.avg_pool2d(lf, kernel_size=2, stride=2)
                ) + self.H2L(self.avg_pool(hf))
            else:
                return self.H2H(hf) + self.upsample(self.L2H(lf)), self.L2L(
                    lf
                ) + self.H2L(self.avg_pool(hf))


class _BatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        alpha_in=0.25,
        alpha_out=0.25,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_BatchNorm2d, self).__init__()
        hf_ch = int(num_features * (1 - alpha_in))
        lf_ch = num_features - hf_ch
        self.bnh = nn.BatchNorm2d(hf_ch)
        self.bnl = nn.BatchNorm2d(lf_ch)

    def forward(self, x):
        hf, lf = x
        return self.bnh(hf), self.bnl(lf)


class _ReLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(_ReLU, self).forward(hf)
        lf = super(_ReLU, self).forward(lf)
        return hf, lf
