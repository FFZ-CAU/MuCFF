"""Evaluation metrics and OOF threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def select_mcc_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    quantile_low: float = 0.02,
    quantile_high: float = 0.98,
    quantile_count: int = 193,
) -> float:
    thresholds = np.unique(
        np.quantile(scores, np.linspace(quantile_low, quantile_high, quantile_count))
    )
    values = [matthews_corrcoef(labels, scores >= threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(values))])


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "threshold": float(threshold),
    }

