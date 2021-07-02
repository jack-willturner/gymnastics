import torch

from abc import ABC, abstractmethod
from typing import Tuple


class Dataset(ABC):
    def __init__(
        self,
        path: str,
        batch_size: int = 128,
        num_workers: int = 0,
        val_split_percentage: float = 0.3,
        seed: int = 1,
        download: bool = False,
    ) -> None:
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.set_transforms()

        train_set, val_set, test_set, _ = self.get_data_sets(
            path, val_split_percentage, seed, download
        )

        train_loader, val_loader, test_loader = self.get_data_loaders(
            train_set, val_set, test_set
        )

        return train_loader, val_loader, test_loader

    @abstractmethod
    def set_transforms(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_data_sets(
        self, path: str, val_split_percentage: float, seed: int, download: bool
    ) -> Tuple[
        torch.utils.data.Dataset,  # train set
        torch.utils.data.Dataset,  # val set
        torch.utils.data.Dataset,  # test set
        int,  # labels
    ]:
        raise NotImplementedError

    def get_data_loaders(self, train_set, val_set, test_set):

        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return self.train_loader, self.val_loader, self.test_loader

    def sample_minibatch(self, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

        if train:
            dataloader = self.train_loader
        else:
            dataloader = self.val_loader

        inputs, classes = next(iter(dataloader))

        return inputs, classes
