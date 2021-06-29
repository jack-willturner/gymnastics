import random
import itertools

import torch.nn as nn

from nasbench import api as API
from nas_101_api.model import Network
from nas_101_api.model_spec import ModelSpec

from .searchspace import SearchSpace
from .utils import Dataset, CIFAR10


class NASBench101SearchSpace(SearchSpace):
    def __init__(self, path_to_api: str = None, dataset: Dataset = CIFAR10) -> None:
        self.dataset = dataset
        self.api = API(path_to_api, verbose=False)

    def sample_random_architecture(self) -> nn.Module:
        arch_id = random.randint(0, len(self) - 1)
        model = self.get_network(arch_id)
        model.arch_id = arch_id
        return model

    def get_accuracy(self, arch_id: int) -> float:
        spec = self.get_spec(arch_id)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = 0.0
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap["final_test_accuracy"]
                if newacc > maxacc:
                    maxacc = newacc
        return maxacc

    def get_accuracy_of_model(self, model: nn.Module) -> float:
        return self.get_accuracy(model.arch_id)

    def get_network(self, arch_id: int) -> nn.Module:
        spec = self.get_spec(arch_id)
        network = Network(spec, self.args)
        return network

    def get_spec(self, unique_hash):
        matrix = self.api.fixed_statistics[unique_hash]["module_adjacency"]
        operations = self.api.fixed_statistics[unique_hash]["module_operations"]
        spec = ModelSpec(matrix, operations)
        return spec

    def __len__(self):
        return len(self.api.hash_iterator())

    def __iter__(self):
        for unique_hash in self.api.hash_iterator():
            network = self.get_network(unique_hash)
            yield unique_hash, network

    def __getitem__(self, index):
        return next(itertools.islice(self.api.hash_iterator(), index, None))
