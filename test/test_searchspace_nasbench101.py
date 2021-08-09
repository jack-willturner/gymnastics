import torch

from gymnastics.benchmarks import get_benchmark

def test_nasbench101_search_space():

    search_space = get_benchmark("NAS-Bench-101", path_to_api="/disk/scratch_ssd/nasbench201/nasbench_only108.tfrecord")
    model = search_space.sample_random_architecture()

    minibatch = torch.rand((10,3,32,32))
    
    y,_ = model(minibatch)
    print("y shape: ", y.size())

    print("accuracy of random model: ", search_space.get_accuracy_of_model(model))
