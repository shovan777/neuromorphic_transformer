import os
import torch
import torchvision
from torch.utils.data import DataLoader
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS
from torchvision import transforms, datasets

def get_dataloaders():
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    test_set = torchvision.datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_set.classes)
    
    return train_loader, test_loader, num_classes, train_set.classes


def get_dataloader_cifar(
    batch_size=128,
    img_size=32,
    validation_split=None,
    use_random_augment=True,
    random_erasing_prob=0.25,
    num_workers=None,
):
    # transform = create_transform(
    #     input_size=img_size,
    #     is_training=True,
    #     color_jitter=0.4,
    #     auto_augment='rand-m9-mstd0.5-inc1',
    #     interpolation='bicubic',
    #     re_prob=0.25,
    #     re_mode='pixel',
    #     re_count=1,
    # )

    train_ops = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_random_augment:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))

    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN, std=STD
            ),
        ]
    )
    if random_erasing_prob > 0:
        train_ops.append(transforms.RandomErasing(p=random_erasing_prob))
    train_transform = transforms.Compose(train_ops)

    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN, std=STD
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_loader = None
    num_workers = (
        max(0, (os.cpu_count() or 0) - 4) if num_workers is None else num_workers
    )

    if validation_split:
        val_size = int(len(train_dataset) * validation_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, val_loader