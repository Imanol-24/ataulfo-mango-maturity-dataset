from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dataloaders(data_dir="data", batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, train_dataset.classes