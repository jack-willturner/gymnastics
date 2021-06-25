import torch

from gymnastics.searchspace.ops import Conv3x3, MixConv, LambdaConv


def test_lambda_conv():

    data = torch.rand((1, 32, 10, 10))
    layer = LambdaConv(32, 32)

    layer(data)


def test_mix_conv():
    data = torch.rand((1, 32, 10, 10))
    layer = MixConv(32, 32, kernel_sizes=[3, 5], groups=2)

    layer(data)


def test_conv3x3():
    data = torch.rand((1, 32, 10, 10))

    layer = Conv3x3(32, 32, kernel_size=3)
    layer(data)
