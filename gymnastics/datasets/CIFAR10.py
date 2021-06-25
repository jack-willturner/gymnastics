import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .dataset import Dataset
from .utils import ImageShape


class CIFAR10Loader(Dataset):
    def __init__(
        self,
        path: str,
        batch_size: int = 128,
        num_workers: int = 0,
        val_split_percentage: float = 0.3,
        seed: int = 1,
        download: bool = False,
    ) -> None:
        super().__init__(
            path, batch_size, num_workers, val_split_percentage, seed, download
        )

    def set_transforms(self) -> None:
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        self.image_shape = ImageShape(3, 32, 32)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.image_shape.width, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_data_sets(
        self, data_filepath, val_set_percentage, random_split_seed, download=False
    ):
        train_set = datasets.CIFAR10(
            root=data_filepath,
            train=True,
            download=download,
            transform=self.transform_train,
        )

        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.CIFAR10(
            root=data_filepath,
            train=False,
            download=download,
            transform=self.transform_validate,
        )

        self.num_labels = 10
        return train_set, val_set, test_set, self.num_labels
