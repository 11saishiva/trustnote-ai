"""
Creates DataLoaders for training, validation and testing.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY
)

from src.dataset import CurrencyDataset
from src.transforms import train_transforms
from src.transforms import val_transforms
from src.transforms import test_transforms


def create_sampler(dataset):

    labels = dataset.get_labels()

    class_counts = np.bincount(labels)

    class_weights = 1.0 / class_counts

    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(

        weights=torch.DoubleTensor(sample_weights),

        num_samples=len(sample_weights),

        replacement=True

    )

    return sampler


def get_dataloaders():

    train_dataset = CurrencyDataset(
        TRAIN_DIR,
        transform=train_transforms
    )

    val_dataset = CurrencyDataset(
        VAL_DIR,
        transform=val_transforms
    )

    test_dataset = CurrencyDataset(
        TEST_DIR,
        transform=test_transforms
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=create_sampler(train_dataset),

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    return (
        train_loader,
        val_loader,
        test_loader
    )