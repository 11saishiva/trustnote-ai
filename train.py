
"""
train.py
Entry point for training the Counterfeit Currency Detection model.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    T_MAX,
    ETA_MIN,
    MODEL_DIR,
)

from src.model import CurrencyClassifier
from src.dataloader import get_dataloaders
from src.trainer import Trainer


def main():

    print("=" * 60)
    print("Counterfeit Currency Detection Training")
    print("=" * 60)

    device = DEVICE

    print(f"Using Device : {device}")

    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train Samples : {len(train_loader.dataset)}")
    print(f"Validation Samples : {len(val_loader.dataset)}")
    print(f"Test Samples : {len(test_loader.dataset)}")

    # Stage 1: Train classifier only
    model = CurrencyClassifier(freeze_backbone=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.trainable_parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_MAX,
        eta_min=ETA_MIN,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=NUM_EPOCHS,
        patience=PATIENCE,
        model_path=MODEL_DIR / "best_model.pth",
        history_path=MODEL_DIR / "training_history.csv",
        use_amp=True,
    )

    trainer.fit()

    print("\nTraining Finished.")
    print(f"Best model saved to: {MODEL_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    main()
