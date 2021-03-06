from .CIFAR10 import CIFAR10Loader

__all__ = ["get_data_loaders"]


def get_data_loaders(dataset_name: str, path_to_dataset: str, batch_size: int = 128):
    if dataset_name == "CIFAR10":
        return CIFAR10Loader(path=path_to_dataset, batch_size=batch_size)
    else:
        raise ValueError("Invalid dataset")
