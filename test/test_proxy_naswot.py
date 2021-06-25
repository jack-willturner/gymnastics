import math
import torch
import torch.nn as nn

from gymnastics.models import ResNet26
from gymnastics.proxies import Proxy, NASWOT
from typing import Dict


def test_proxy_naswot():
    from gymnastics.genotypes import generate_resnet_genotypes

    generated_configs = generate_resnet_genotypes()
    generated_config = generated_configs["resnet26"]

    model: nn.Module = ResNet26(generated_config)

    minibatch: torch.Tensor = torch.rand(64, 3, 32, 32)

    naswot: Proxy = NASWOT()
    score: float = naswot.score(model, minibatch)

    assert not math.isnan(score)
