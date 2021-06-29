from gymnastics.searchspace import SearchSpace
from gymnastics.searchspace.utils import Dataset, CIFAR10


class NASBench301SearchSpace(SearchSpace):
    def __init__(self, path_to_api: str = None, dataset: Dataset = CIFAR10) -> None:
        raise NotImplementedError(
            "NASBench301 not implemented yet. Waiting for https://github.com/automl/nasbench301/issues/3 to be resolved."
        )
