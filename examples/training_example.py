import argparse
import os
import torch
import torch.nn as nn
import yaml

from gymnastics.searchspace.searchspace import SearchSpace
from gymnastics.datasets import get_data_loaders
from gymnastics.benchmarks import get_benchmark
from gymnastics.proxies import NASWOT, Proxy
from gymnastics.training import full_training_run

parser = argparse.ArgumentParser(
    description="Evaluate a proxy on various NAS-Benchmarks"
)

parser.add_argument(
    "--path_to_api",
    default="/disk/scratch_ssd/nasbench201/NASBench_v1_1.pth",
    type=str,
    help="Path to nas-bench api file",
)
parser.add_argument(
    "--path_to_data",
    default="/disk/scratch_ssd/nasbench201/cifar10",
    type=str,
    help="Path to actual dataset",
)


args = parser.parse_args()

experiment_name = os.path.split("experiment_config/nasbench_201.yaml")[1].split(".")[0]


def get_best_model_using_proxy(proxy, search_space, train_loader, num_samples):

    best_score: float = 0.0

    for _ in range(num_samples):
        minibatch: torch.Tensor = train_loader.sample_minibatch()
        model: nn.Module = search_space.sample_random_architecture()

        score: float = proxy.score(model, minibatch)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


if __name__ == "main":

    # load the config file
    with open(args.experiment_config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # experimental setup
    proxy: Proxy = NASWOT()
    search_space: SearchSpace = get_benchmark(
        "NASBench201", path_to_api=args.path_to_api
    )
    train_loader, _, _ = get_data_loaders("CIFAR10", path_to_dataset=args.path_to_data)

    model = get_best_model_using_proxy(
        proxy, search_space, train_loader, args.num_samples
    )

    accuracy = full_training_run(model)

    print(accuracy)
