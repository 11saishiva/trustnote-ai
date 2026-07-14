"""
Model checkpoint utility.
"""

from pathlib import Path
import torch


class ModelCheckpoint:

    def __init__(self, filepath):

        self.best_f1 = -1

        self.filepath = Path(filepath)

        self.filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, model, f1):

        if f1 > self.best_f1:

            self.best_f1 = f1

            torch.save(
                model.state_dict(),
                self.filepath
            )

            print(
                f"\nSaved Best Model "
                f"(F1 = {f1:.4f})"
            )