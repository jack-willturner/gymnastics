import torch
import torch.nn as nn
from gymnastics.proxies.proxy import Proxy


class Params(Proxy):
    def score(self, model: nn.Module, minibatch: torch.Tensor) -> float:

        raise NotImplementedError("param scoring")
