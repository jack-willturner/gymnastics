from .conv import Conv
from .gconv import GConv
from .dwseparable import DWSeparableConv
from .lambdaconv import LambdaConv
from .mixconv import MixConv
from .octaveconv import OctConv, _BatchNorm2d, _ReLU
from .shiftconv import ShiftConv

from torch.nn import BatchNorm2d, ReLU

from typing import Callable

layer_type_registry = [
    "Conv",
    "GConv",
    "DWSeparableConv",
    "LambdaConv",
    "MixConv",
    "ShiftConv",
]


def get_conv_bn_relu(conv: str) -> Callable:
    if conv == "Conv":
        return Conv, BatchNorm2d, ReLU
    elif conv == "GConv":
        return GConv, BatchNorm2d, ReLU
    elif conv == "DWSeparableConv":
        return DWSeparableConv, BatchNorm2d, ReLU
    elif conv == "MixConv":
        return MixConv, BatchNorm2d, ReLU
    elif conv == "LambdaConv":
        return LambdaConv, BatchNorm2d, ReLU
    elif conv == "OctaveConv":
        return OctConv, _BatchNorm2d, _ReLU
    elif conv == "ShiftConv":
        return ShiftConv, BatchNorm2d, ReLU
    else:
        raise NotImplementedError(f"Unrecognised type of convolution {conv}")
