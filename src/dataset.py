"""
PyTorch Dataset for Counterfeit Currency Detection.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CurrencyDataset(Dataset):

    LABEL_MAP = {
        "real": 0,
        "fake": 1
    }

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []

        for class_name, label in self.LABEL_MAP.items():

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for denomination in sorted(class_dir.iterdir()):

                if not denomination.is_dir():
                    continue

                for image_path in sorted(denomination.iterdir()):

                    if image_path.suffix.lower() not in {
                        ".jpg",
                        ".jpeg",
                        ".png"
                    }:
                        continue

                    self.samples.append({

                        "path": image_path,

                        "label": label,

                        "class": class_name,

                        "denomination": denomination.name

                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, sample["label"]

    def get_labels(self):
        return [sample["label"] for sample in self.samples]

    def class_distribution(self):

        dist = {
            "REAL": 0,
            "FAKE": 0
        }

        for sample in self.samples:

            if sample["label"] == 0:
                dist["REAL"] += 1
            else:
                dist["FAKE"] += 1

        return dist