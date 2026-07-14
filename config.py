from pathlib import Path

# ======================================
# Project Paths
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR = PROJECT_ROOT / "models"

# ======================================
# Dataset Split
# ======================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_STATE = 42

# ======================================
# Images
# ======================================

IMAGE_SIZE = 224

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

# ======================================
# Model
# ======================================

NUM_CLASSES = 2

CLASS_NAMES = [
    "REAL",
    "FAKE"
]

MODEL_NAME = "mobilenet_v3_large"

PRETRAINED = True

# ======================================
# Training
# ======================================

BATCH_SIZE = 32

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PATIENCE = 5

# ======================================
# DataLoader
# ======================================


PIN_MEMORY = True

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# ==========================
# Training
# ==========================

# ===============================
# Device
# ===============================

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ===============================
# Training
# ===============================



GRADIENT_CLIP = 1.0

USE_AMP = torch.cuda.is_available()

# ===============================
# Scheduler
# ===============================

T_MAX = NUM_EPOCHS

ETA_MIN = 1e-6