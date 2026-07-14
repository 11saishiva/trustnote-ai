
"""
trainer.py
Production-ready trainer for Counterfeit Currency Detection
"""

from pathlib import Path
import csv
import time

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.metrics import calculate_metrics
from src.checkpoint import ModelCheckpoint


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        patience,
        model_path,
        gradient_clip=1.0,
        use_amp=True,
        history_path="models/training_history.csv",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.gradient_clip = gradient_clip

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.checkpoint = ModelCheckpoint(model_path)

        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self.history = []

    def train_one_epoch(self):
        self.model.train()

        running_loss = 0.0

        y_true = []
        y_pred = []
        y_prob = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in pbar:

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        return running_loss / len(self.train_loader), metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        running_loss = 0.0

        y_true = []
        y_pred = []
        y_prob = []

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        for images, labels in pbar:

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        return running_loss / len(self.val_loader), metrics

    def _save_history(self):
        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
            "train_f1",
            "val_f1"
        ]

        with open(self.history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for row in self.history:
                writer.writerow(row)

    def fit(self):

        best_f1 = -1.0
        patience_counter = 0

        print("\nStarting Training...\n")

        start = time.time()

        for epoch in range(self.epochs):

            train_loss, train_metrics = self.train_one_epoch()

            val_loss, val_metrics = self.validate()

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:
                    self.scheduler.step(val_loss)

            self.history.append([
                epoch + 1,
                train_loss,
                val_loss,
                train_metrics["accuracy"],
                val_metrics["accuracy"],
                train_metrics["f1"],
                val_metrics["f1"]
            ])

            print(
                f"\nEpoch {epoch+1}/{self.epochs}"
                f"\nTrain Loss : {train_loss:.4f}"
                f"\nVal Loss   : {val_loss:.4f}"
                f"\nTrain Acc  : {train_metrics['accuracy']:.4f}"
                f"\nVal Acc    : {val_metrics['accuracy']:.4f}"
                f"\nTrain F1   : {train_metrics['f1']:.4f}"
                f"\nVal F1     : {val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_f1:

                best_f1 = val_metrics["f1"]

                patience_counter = 0

                self.checkpoint.save(
                    self.model,
                    best_f1
                )

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_f1": best_f1,
                }, Path(self.checkpoint.filepath).with_suffix(".ckpt"))

            else:

                patience_counter += 1

            if patience_counter >= self.patience:

                print("\nEarly stopping triggered.")

                break

        self._save_history()

        elapsed = time.time() - start

        print(f"\nTraining completed in {elapsed/60:.2f} minutes.")
        print(f"Best Validation F1 : {best_f1:.4f}")
