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
    # generate a random network
    genotype: Dict = generate_baseline_genotype_with_random_middle_convs(
        BottleneckGenerator(), n_blocks=[2, 2, 2, 2], strides=[1, 2, 2, 2]
    )

    model: nn.Module = ResNet26(genotype)

    minibatch: torch.Tensor = torch.rand(64, 3, 32, 32)

    naswot: Proxy = NASWOT()
    score: float = naswot.score(model, minibatch)

    assert not math.isnan(score)
