"""
Image transformations for Currency Detection.

Training:
    - Realistic augmentations to improve generalization.

Validation/Test:
    - Deterministic preprocessing only.
"""

from torchvision import transforms

from config import IMAGE_SIZE

# ImageNet normalization values (used by pretrained MobileNetV3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================================
# Training Transforms
# ==========================================================

train_transforms = transforms.Compose([

    # Resize slightly larger before cropping
    transforms.Resize((256, 256)),

    # Simulate different camera distances
    transforms.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(0.85, 1.0),
        ratio=(0.95, 1.05)
    ),

    # Small camera tilt
    transforms.RandomRotation(
        degrees=5
    ),

    # Small translation / zoom / shear
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        shear=2
    ),

    # Simulate angled photographs
    transforms.RandomPerspective(
        distortion_scale=0.15,
        p=0.30
    ),

    # Different lighting conditions
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.05,
        hue=0.02
    ),

    # Slight blur from camera movement
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.20),

    transforms.ToTensor(),

    # Simulate small occlusions (finger, shadow, dust)
    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.08),
        ratio=(0.3, 3.3),
        value="random"
    ),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])

# ==========================================================
# Validation / Test Transforms
# ==========================================================

val_transforms = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])

# Use the same preprocessing for testing
test_transforms = val_transforms