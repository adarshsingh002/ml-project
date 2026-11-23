"""
evaluate.py — Model evaluation helpers.

Computes metrics on the held-out test set and produces
JSON-serializable results for the Streamlit dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.features import CLASS_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Full evaluation dict for a 3-class classifier.

    Returns:
        Dictionary with accuracy, per-class + aggregate metrics,
        confusion matrix, ROC-AUC, and class distribution.
    """
    acc = float(accuracy_score(y_true, y_pred))

    # Per-class precision, recall, f1
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0,
    )
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]), 4),
            "f1_score":  round(float(f1[i]), 4),
            "support":   int(support[i]),
        }

    # Macro & weighted averages
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )

    # ROC-AUC (one-vs-rest, macro)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # Class distribution in ground truth
    unique, counts = np.unique(y_true, return_counts=True)
    distribution = {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}

    return {
        "accuracy": round(acc, 4),
        "roc_auc_macro": round(roc_auc, 4) if roc_auc is not None else None,
        "per_class": per_class,
        "macro_avg": {
            "precision": round(float(prec_m), 4),
            "recall":    round(float(rec_m), 4),
            "f1_score":  round(float(f1_m), 4),
        },
        "weighted_avg": {
            "precision": round(float(prec_w), 4),
            "recall":    round(float(rec_w), 4),
            "f1_score":  round(float(f1_w), 4),
        },
        "confusion_matrix": cm,
        "class_distribution": distribution,
        "total_samples": int(len(y_true)),
    }


def save_metrics(metrics: dict, path: str = "models/metrics.json") -> None:
    """Persist metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📊 Metrics saved to {path}")


def load_metrics(path: str = "models/metrics.json") -> dict:
    """Load metrics from JSON."""
    with open(path) as f:
        return json.load(f)


def print_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print sklearn classification report to stdout."""
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    ))
