from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR
from config import TRAIN_RATIO
from config import VAL_RATIO
from config import TEST_RATIO
from config import RANDOM_STATE
from config import VALID_IMAGE_EXTENSIONS

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# ---------------------------------------

output_dirs = {
    "train": DATA_DIR / "train",
    "val": DATA_DIR / "val",
    "test": DATA_DIR / "test"
}

for folder in output_dirs.values():
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------

for cls in ["real", "fake"]:

    class_dir = DATA_DIR / cls

    if not class_dir.exists():
        continue

    for denomination in sorted(class_dir.iterdir()):

        if not denomination.is_dir():
            continue

        images = [

            x for x in denomination.iterdir()

            if x.suffix.lower() in VALID_IMAGE_EXTENSIONS

        ]

        train_imgs, temp_imgs = train_test_split(

            images,

            train_size=TRAIN_RATIO,

            random_state=RANDOM_STATE,

            shuffle=True

        )

        val_imgs, test_imgs = train_test_split(

            temp_imgs,

            test_size=0.5,

            random_state=RANDOM_STATE,

            shuffle=True

        )

        splits = {

            "train": train_imgs,

            "val": val_imgs,

            "test": test_imgs

        }

        for split_name, split_images in splits.items():

            dest = (

                output_dirs[split_name]

                / cls

                / denomination.name

            )

            dest.mkdir(parents=True, exist_ok=True)

            for img in split_images:

                shutil.copy2(

                    img,

                    dest / img.name

                )

print()

print("=" * 50)

print("Dataset Successfully Split")

print("=" * 50)

print()

print("Train :", TRAIN_RATIO)

print("Validation :", VAL_RATIO)

print("Test :", TEST_RATIO)