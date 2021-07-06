import yaml
import argparse

import pandas as pd
from tabulate import tabulate

import torch
import torch.nn as nn

from gymnastics.datasets import get_data_loaders
from gymnastics.benchmarks import get_benchmark
from gymnastics.proxies import get_proxy

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

parser.add_argument(
    "--path_to_results",
    default="results/",
    type=str,
    help="The folder in which I should store the results file(s)",
)

args = parser.parse_args()

results = []

with open(args.experiment_config, "r") as file:
    experiment_config = yaml.safe_load(file)

proxy = get_proxy(experiment_config["proxy"])

for benchmark in experiment_config["benchmarks"]:

    search_space = get_benchmark(
        benchmark["name"], path_to_api=benchmark["path_to_api"]
    )

    train_loader, _, _ = get_data_loaders(
        benchmark["dataset"], benchmark["path_to_dataset"]
    )

    for _ in range(experiment_config["num_trials"]):

        best_score: float = 0.0

        for _ in range(experiment_config["num_samples"]):
            minibatch: torch.Tensor = train_loader.sample_minibatch()
            model: nn.Module = search_space.sample_random_architecture()

            score: float = proxy.score(model, minibatch)

            if score > best_score:
                best_score = score
                best_model = model

        results.append(
            [
                benchmark["name"],
                benchmark["dataset"],
                best_model.arch_id,
                experiment_config["proxy"],
                experiment_config["num_samples"],
                score,
                search_space.get_accuracy_of_model(best_model),
            ]
        )

results = pd.DataFrame(
    results,
    columns=[
        "Benchmark",
        "Dataset",
        "Arch ID",
        "Proxy",
        "Number of Samples",
        "Score",
        "Accuracy",
    ],
)

print(tabulate(results, headers="keys", tablefmt="psql"))

results.to_pickle(f"{args.path_to_results}/{args.experiment_config}.pd")
