import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
    return transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    # Download training data
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    # Download test data (for validation)
    val_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
