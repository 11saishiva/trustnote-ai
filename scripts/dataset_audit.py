from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR
from config import REPORT_DIR
from config import VALID_IMAGE_EXTENSIONS

from src.utils import sha256

REPORT_DIR.mkdir(exist_ok=True)

# ============================================
# Variables
# ============================================

class_counter = Counter()

denomination_counter = defaultdict(Counter)

image_sizes = Counter()

image_modes = Counter()

image_formats = Counter()

duplicates = defaultdict(list)

corrupted = []

rows = []

total_images = 0

# ============================================
# Scan Dataset
# ============================================

print("\nScanning Dataset...\n")

for cls in ["real", "fake"]:

    class_dir = DATA_DIR / cls

    if not class_dir.exists():

        print(f"Missing Folder : {class_dir}")

        continue

    for denomination in sorted(class_dir.iterdir()):

        if not denomination.is_dir():
            continue

        images = list(denomination.iterdir())

        for image_path in tqdm(
                images,
                desc=f"{cls}/{denomination.name}"
        ):

            if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue

            total_images += 1

            class_counter[cls] += 1

            denomination_counter[cls][denomination.name] += 1

            file_hash = sha256(image_path)

            duplicates[file_hash].append(str(image_path))

            try:

                img = Image.open(image_path)

                img.verify()

                img = Image.open(image_path)

                image_sizes[img.size] += 1

                image_modes[img.mode] += 1

                image_formats[img.format] += 1

                rows.append({

                    "path": str(image_path),

                    "class": cls,

                    "denomination": denomination.name,

                    "width": img.width,

                    "height": img.height,

                    "mode": img.mode,

                    "format": img.format

                })

            except Exception:

                corrupted.append(str(image_path))

# ============================================
# Duplicate List
# ============================================

duplicate_files = []

for h, files in duplicates.items():

    if len(files) > 1:

        duplicate_files.extend(files)

# ============================================
# Print Report
# ============================================

print("\n")

print("=" * 60)

print("DATASET SUMMARY")

print("=" * 60)

print(f"Total Images : {total_images}")

print(f"Real Images  : {class_counter['real']}")

print(f"Fake Images  : {class_counter['fake']}")

print()

print("REAL")

for d in sorted(denomination_counter["real"], key=int):

    print(f"{d:>5} : {denomination_counter['real'][d]}")

print()

print("FAKE")

for d in sorted(denomination_counter["fake"], key=int):

    print(f"{d:>5} : {denomination_counter['fake'][d]}")

print()

print("Image Modes")

for mode, count in image_modes.items():

    print(mode, count)

print()

print("Formats")

for fmt, count in image_formats.items():

    print(fmt, count)

print()

print(f"Corrupted Images : {len(corrupted)}")

print(f"Duplicate Images : {len(duplicate_files)}")

# ============================================
# Save CSV Reports
# ============================================

pd.DataFrame(rows).to_csv(

    REPORT_DIR / "dataset_report.csv",

    index=False

)

pd.DataFrame({

    "corrupted": corrupted

}).to_csv(

    REPORT_DIR / "corrupted_images.csv",

    index=False

)

pd.DataFrame({

    "duplicates": duplicate_files

}).to_csv(

    REPORT_DIR / "duplicate_images.csv",

    index=False

)

print()

print("Reports saved inside reports/")