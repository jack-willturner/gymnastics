import yaml


def test_resnet18_genotype_generator():

    import os

    absolute_path = os.path.abspath(__file__)

    with open("test/genotypes/resnet18_baseline.yaml", "r") as config_file:
        ground_truth_config = yaml.safe_load(config_file)

    from genotypes import generate_resnet_genotypes

    generated_configs = generate_resnet_genotypes()
    generated_config = generated_configs["resnet18"]

    assert ground_truth_config == generated_config


def test_random_resnet_generator():
    import torch
    import torch.nn as nn
    from genotypes import (
        BottleneckGenerator,
        generate_baseline_genotype_with_random_middle_convs,
    )
    from models import ResNet26
    from typing import Dict

    for _ in range(10):

        genotype: Dict = generate_baseline_genotype_with_random_middle_convs(
            BottleneckGenerator(), n_blocks=[2, 2, 2, 2], strides=[1, 2, 2, 2]
        )

        model: nn.Module = ResNet26(genotype)

        minibatch: torch.Tensor = torch.rand(64, 3, 32, 32)

        _ = model(minibatch)
