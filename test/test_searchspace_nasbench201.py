import torch

from gymnastics.benchmarks import NASBench201SearchSpace

def test_nasbench201_search_space():
    search_space = NASBench201SearchSpace(path_to_api="/disk/scratch_ssd/nasbench201/NASBench_v1_1.pth")
    model = search_space.sample_random_architecture()

    minibatch = torch.rand((10,3,32,32))
    
    y = model(minibatch)
    print(y.size())
