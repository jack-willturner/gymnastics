# nas-bench-201
from torch.nn.modules.pooling import MaxPool2d
from .avgpool2d import AvgPool2d
from .conv1x1 import Conv1x1
from .conv3x3 import Conv3x3
from .identity import Identity
from .zeroize import Zeroize
from .maxpool2d import MaxPool2d
from .dwseparable import DWSeparableConv
from .gconv import GConv
from .lambdaconv import LambdaConv
from .mixconv import MixConv
from .octaveconv import OctConv
from .shiftconv import ShiftConv


__all__ = [
    "AvgPool2d",
    "MaxPool2d",
    "Conv1x1",
    "Conv3x3",
    "Identity",
    "Zeroize",
    "DWSeparableConv",
    "GConv",
    "LambdaConv",
    "MixConv",
    "OctConv",
    "ShiftConv",
]
