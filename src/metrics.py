"""
Evaluation metrics.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        labels,
        predictions
    )

    metrics["precision"] = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    metrics["recall"] = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    metrics["f1"] = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    try:

        metrics["roc_auc"] = roc_auc_score(
            labels,
            probabilities
        )

    except Exception:

        metrics["roc_auc"] = 0.0

    return metrics