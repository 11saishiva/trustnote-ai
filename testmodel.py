import torch

from src.model import CurrencyClassifier

model = CurrencyClassifier()

dummy = torch.randn(2, 3, 224, 224)

output = model(dummy)

print("Output Shape:", output.shape)