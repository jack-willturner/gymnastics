import argparse
import os
import torch
import torch.nn as nn

from gymnastics.searchspace.searchspace import SearchSpace
from gymnastics.datasets import CIFAR10Loader
from gymnastics.benchmarks import NASBench201SearchSpace
from gymnastics.proxies import NASWOT, Proxy
from sacred import Experiment

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

parser.add_argument(
    "--experiment_config",
    default="experiment_configs/nasbench_201.yaml",
    type=str,
    help="Needs to contain: n_trials, n_samples, proxy",
)

args = parser.parse_args()

experiment_name = os.path.split("experiment_config/nasbench_201.yaml")[1].split(".")[0]

ex = Experiment(experiment_name)
ex.add_config(args.experiment_config)


@ex.automain
def run_experiment():

    accuracies = []

    for _ in range(args.num_trials):
        search_space: SearchSpace = NASBench201SearchSpace(path_to_api=args.path_to_api)

        train_loader, _ = CIFAR10Loader(path=args.path_to_data)

        proxy: Proxy = NASWOT()

        best_score: float = 0.0

        for _ in range(args.num_samples):
            minibatch: torch.Tensor = train_loader.sample_minibatch()
            model: nn.Module = search_space.sample_random_architecture()

            score: float = proxy.score(model, minibatch)

            if score > best_score:
                best_score = score
                best_model = model

        accuracies.append(search_space.get_accuracy_of_model(best_model))

    return accuracies
