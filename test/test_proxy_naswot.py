import math
import torch
import torch.nn as nn

from genotypes import (
    BottleneckGenerator,
    generate_baseline_genotype_with_random_middle_convs,
)
from models import ResNet26
from proxies import Proxy, NASWOT
from typing import Dict


def test_proxy_naswot():
    from genotypes import generate_resnet_genotypes

    generated_configs = generate_resnet_genotypes()
    generated_config = generated_configs["resnet26"]

    model: nn.Module = ResNet26(generated_config)

    minibatch: torch.Tensor = torch.rand(64, 3, 32, 32)

    naswot: Proxy = NASWOT()
    score: float = naswot.score(model, minibatch)

    assert not math.isnan(score)
