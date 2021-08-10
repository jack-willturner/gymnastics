import random
import itertools

import torch.nn as nn

from types import SimpleNamespace

from nasbench import api as API
from .nas_101_api.model import Network
from .nas_101_api.model_spec import ModelSpec

from gymnastics.searchspace import SearchSpace
from gymnastics.searchspace.utils import Dataset, CIFAR10


class NASBench101SearchSpace(SearchSpace):
    def __init__(self, path_to_api: str = None, dataset: Dataset = CIFAR10) -> None:
        self.dataset = dataset
        self.api = API.NASBench(path_to_api)

    def sample_random_architecture(self, num_classes: int = None) -> nn.Module:
        arch_id = random.randint(0, len(self) - 1)
        if num_classes is None:
            num_classes = self.dataset.num_classes
        model = self.get_network(arch_id, num_classes)
        model.arch_id = arch_id
        return model

    def get_accuracy(self, arch_id: int) -> float:
        unique_hash = self.convert_arch_id_to_unique_hash(arch_id)
        spec = self.get_spec(unique_hash)
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

    def convert_arch_id_to_unique_hash(self, arch_id: int) -> str:
        return next(itertools.islice(self.api.hash_iterator(), arch_id, None))

    def get_network(self, arch_id: int, num_labels: int = 1) -> nn.Module:
        # convert arch_id from int to hash
        unique_hash = self.convert_arch_id_to_unique_hash(arch_id)
        # fetch spec from API
        spec = self.get_spec(unique_hash)

        # build an args dict
        args = {
            "num_labels": num_labels,
            "num_stacks": 3,
            "num_modules_per_stack": 3,
            "stem_out_channels": 16,
        }  # nasbench wants this in namespace format
        args = SimpleNamespace(**args)

        network = Network(spec, args=args)
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
