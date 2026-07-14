
"""
evaluate.py
Evaluate trained model on the test dataset.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from config import DEVICE, MODEL_DIR
from src.model import CurrencyClassifier
from src.dataloader import get_dataloaders


def main():
    _, _, test_loader = get_dataloaders()

    model = CurrencyClassifier()
    state = torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    print("\n===== Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)

    with open(MODEL_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["REAL", "FAKE"]
    )

    with open(MODEL_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    print("\nClassification Report\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["REAL", "FAKE"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png")
    plt.close(fig)

    print("\nSaved:")
    print(" - test_metrics.json")
    print(" - classification_report.txt")
    print(" - confusion_matrix.png")


if __name__ == "__main__":
    main()
