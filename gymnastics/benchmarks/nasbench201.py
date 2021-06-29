import random
import pandas as pd
import torch.nn as nn

from .nas_201_models import get_cell_based_tiny_net

from gymnastics.searchspace import SearchSpace
from gymnastics.searchspace.utils import Dataset, CIFAR10
from typing import Tuple


class NASBench201SearchSpace(SearchSpace):
    def __init__(self, path_to_api: str = None, dataset: Dataset = CIFAR10) -> None:
        self.dataset = dataset
        self.api = pd.read_pickle(path_to_api)

    def sample_random_architecture(self) -> nn.Module:
        arch_id = random.randint(0, len(self) - 1)
        model = self.get_network(arch_id)
        model.arch_id = arch_id
        return model

    def get_network(self, arch_id: int) -> nn.Module:
        config = self.api.get_net_config(arch_id, "cifar10-valid")
        config["num_classes"] = self.dataset.num_classes
        network = get_cell_based_tiny_net(config)
        return network

    def get_accuracy_of_model(
        self, model: nn.Module
    ) -> Tuple[float, float, float, float, float, float]:

        archinfo = self.api.query_meta_info_by_index(model.arch_id, hp="200")

        c10 = archinfo.get_metrics("cifar10", "ori-test")["accuracy"]
        c10_val = archinfo.get_metrics("cifar10-valid", "x-valid")["accuracy"]

        c100 = archinfo.get_metrics("cifar100", "x-test")["accuracy"]
        c100_val = archinfo.get_metrics("cifar100", "x-valid")["accuracy"]

        imagenet = archinfo.get_metrics("ImageNet16-120", "x-test")["accuracy"]
        imagenet_val = archinfo.get_metrics("ImageNet16-120", "x-valid")["accuracy"]

        return c10, c10_val, c100, c100_val, imagenet, imagenet_val

    def __len__(self) -> int:
        return 15625

    def __iter__(self):
        for arch_id in range(len(self)):
            network = self.get_network(arch_id)
            yield arch_id, network

    def __getitem__(self, index):
        return index
