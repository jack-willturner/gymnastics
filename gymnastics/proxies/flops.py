import torch
import torch.nn as nn
from gymnastics.proxies.proxy import Proxy


class Flops(Proxy):
    def score(self, model: nn.Module, minibatch: torch.Tensor) -> float:

        return 0
