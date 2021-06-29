import torch
from gymnastics.benchmarks import NDSSearchSpace


def test_nds_resnet():
    searchspace = NDSSearchSpace(
        "/Users/jackturner/work/nds/data/ResNet.json", searchspace="ResNet"
    )
    data = torch.rand((1, 3, 32, 32))
    for _ in range(10):
        model = searchspace.sample_random_architecture()
        y, _ = model(data)
        print(y.size())

        print(searchspace.get_accuracy_of_model(model))
