import torch
import numpy as np
import torch.nn as nn
from gymnastics.proxies.proxy import Proxy


class Params(Proxy):
    def score(self, model: nn.Module, minibatch: torch.Tensor) -> float:

        return 0
