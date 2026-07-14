
"""
inference.py
Production inference for Counterfeit Currency Detection.
"""

from pathlib import Path
import json
import torch
import torch.nn.functional as F
from PIL import Image

from config import IMAGE_SIZE
from src.model import CurrencyClassifier
from src.transforms import test_transforms


class CurrencyDetector:
    def __init__(self, model_path="models/best_model.pth", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = CurrencyClassifier()
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        self.classes = {0: "REAL", 1: "FAKE"}

    def preprocess(self, image_path):
        image = Image.open(Path(image_path)).convert("RGB")
        tensor = test_transforms(image)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_path):
        x = self.preprocess(image_path)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return self.classes[pred.item()], conf.item()

    @torch.no_grad()
    def predict_with_details(self, image_path):
        x = self.preprocess(image_path)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))
        return {
            "prediction": self.classes[pred],
            "confidence": float(probs[pred]),
            "probabilities": {
                "REAL": float(probs[0]),
                "FAKE": float(probs[1]),
            },
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/best_model.pth")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    detector = CurrencyDetector(args.model)

    if args.json:
        print(json.dumps(detector.predict_with_details(args.image), indent=4))
    else:
        pred, conf = detector.predict(args.image)
        print(f"Prediction : {pred}")
        print(f"Confidence : {conf:.4f}")
