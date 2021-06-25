import torch
import yaml

from gymnastics.models import ResNet18


def test_resnet18():

    with open("gymnastics/genotypes/resnet18.yaml", "r") as config_file:
        configs = yaml.safe_load(config_file)

    net = ResNet18(configs)

    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
