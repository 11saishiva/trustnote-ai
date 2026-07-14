"""
MobileNetV3-Large model for Counterfeit Indian Currency Detection.

Supports:
- ImageNet pretrained weights
- Transfer learning
- Optional backbone freezing
- Binary classification (REAL / FAKE)
"""

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES, PRETRAINED


class CurrencyClassifier(nn.Module):
    """
    MobileNetV3-Large classifier for binary image classification.
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()

        # -------------------------------------------------------
        # Load pretrained MobileNetV3-Large
        # -------------------------------------------------------

        if PRETRAINED:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None

        self.model = models.mobilenet_v3_large(weights=weights)

        # -------------------------------------------------------
        # Freeze feature extractor (optional)
        # -------------------------------------------------------

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # -------------------------------------------------------
        # Get classifier input dimension
        # -------------------------------------------------------

        # First Linear layer receives 960 features
        in_features = self.model.classifier[0].in_features

        # -------------------------------------------------------
        # Replace classifier
        # -------------------------------------------------------

        self.model.classifier = nn.Sequential(

            nn.Linear(in_features, 512),

            nn.BatchNorm1d(512),

            nn.Hardswish(),

            nn.Dropout(0.30),

            nn.Linear(512, 128),

            nn.BatchNorm1d(128),

            nn.Hardswish(),

            nn.Dropout(0.20),

            nn.Linear(128, NUM_CLASSES)

        )

    def forward(self, x):
        return self.model(x)

    def unfreeze_backbone(self):
        """
        Unfreeze the entire backbone for fine-tuning.
        """

        for param in self.model.features.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """
        Freeze the backbone.
        """

        for param in self.model.features.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        """
        Returns only trainable parameters.
        Useful when creating the optimizer.
        """

        return filter(lambda p: p.requires_grad, self.parameters())